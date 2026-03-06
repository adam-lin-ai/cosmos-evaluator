# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to set up ONNX runtime."""

import gc
import logging
import warnings

import onnxruntime as ort

logger = logging.getLogger(__name__)


def clear_gpu_memory() -> None:
    """Clear GPU memory caches to reduce fragmentation and free unused memory.

    This function should be called between heavy GPU operations (e.g., between
    running different models) to consolidate GPU memory and prevent OOM errors
    from memory fragmentation.

    The function:
    1. Runs Python garbage collection to free unreferenced objects
    2. Clears PyTorch's CUDA memory cache (if available)
    3. Synchronizes CUDA streams to ensure all operations are complete
    """
    # Run garbage collection to free unreferenced Python objects
    gc.collect()

    # Clear PyTorch CUDA cache if torch is available
    try:
        import torch

        # Defensive check: ensure torch is not None and has cuda attribute
        if torch is None or not hasattr(torch, "cuda"):
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared PyTorch CUDA cache")
    except ImportError:
        # PyTorch not available, skip CUDA cache clearing
        pass
    except Exception as e:
        # Best-effort cleanup; log and continue rather than failing the caller
        logger.warning("Failed to clear CUDA cache during clear_gpu_memory: %s", e)


def configure_onnx_logging(verbose: bool = False):
    """Configure ONNX runtime logging to reduce warnings.

    Args:
        verbose: Enable verbose logging if True
    """
    # Suppress ONNX runtime warnings unless verbose mode is enabled
    if not verbose:
        # Set ONNX runtime logging to only show errors
        ort.set_default_logger_severity(3)  # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal

        # Suppress specific ONNX warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")


def get_inference_session(model_path: str, model_device: str = "cuda", verbose: bool = False):
    """Takes as input an ONNX model path and returns an ORT inference session.

    Args:
        model_path: path to ONNX file.
        model_device: Device to run the model on ("cuda" or "cpu")
        verbose: Enable verbose logging

    Returns:
        Onnx inference session.
    """
    # Configure providers with performance-optimized settings
    providers = ["CPUExecutionProvider"]

    if model_device == "cuda":
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    # Use kSameAsRequested to allocate exactly what's needed, avoiding power-of-2 jumps
                    # that can cause the arena to get stuck at certain sizes
                    "arena_extend_strategy": "kSameAsRequested",
                    # Set high GPU memory limit to allow arena to grow as needed
                    "gpu_mem_limit": 16 * 1024 * 1024 * 1024,  # 16GB (generous for 24GB GPU)
                    # Use HEURISTIC instead of EXHAUSTIVE to reduce memory pressure from cuDNN workspace
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv1d_pad_to_nc1d": True,  # Optimize 1D convolutions
                    "do_copy_in_default_stream": True,  # Better memory management
                    "enable_cuda_graph": False,  # Disable to avoid shape massaging conflicts
                    # Reduce cuDNN workspace memory usage
                    "cudnn_conv_use_max_workspace": False,
                },
            )
        ] + providers

    sess_options = ort.SessionOptions()

    # Reduce logging verbosity unless explicitly requested
    if verbose:
        sess_options.log_severity_level = 0  # Verbose logging
    else:
        sess_options.log_severity_level = 2  # Only warnings and errors

    # Performance optimized settings
    sess_options.enable_cpu_mem_arena = True  # Enable CPU memory arena for better memory management
    sess_options.enable_mem_pattern = True
    sess_options.intra_op_num_threads = 0  # Use all available CPU cores (0 = auto-detect)
    sess_options.inter_op_num_threads = 0  # Use all available CPU cores for parallel ops
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_profiling = False  # Disable profiling for better performance

    try:
        ort.preload_dlls()
        ort_session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)

        # Always log which provider is being used
        active_providers = ort_session.get_providers()
        # Use print to ensure visibility regardless of logging config
        print(
            f"[ONNX] Providers: {active_providers}, GPU mem limit: {providers[0][1].get('gpu_mem_limit', 'N/A') if isinstance(providers[0], tuple) else 'N/A'}"
        )
        logger.info(f"ONNX Runtime using providers: {active_providers}")

        if verbose:
            logger.info("ORT device: {}".format(ort.get_device()))

            model_inputs = ort_session.get_inputs()
            for i, model_input in enumerate(model_inputs):
                input_name = model_input.name
                input_shape = model_input.shape
                logger.info("Model input{} name: {}, shape: {}".format(i, input_name, input_shape))

            model_outputs = ort_session.get_outputs()
            for i, model_output in enumerate(model_outputs):
                output_name = model_output.name
                output_shape = model_output.shape
                logger.info("Model output{} name: {}, shape: {}".format(i, output_name, output_shape))

        return ort_session

    except Exception as e:
        logger.error(f"Failed to create ONNX inference session: {e}")
        # Fallback to CPU-only if CUDA fails
        logger.info("Falling back to CPU-only execution")
        providers = ["CPUExecutionProvider"]
        sess_options.log_severity_level = 2  # Reduce logging in fallback
        return ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
