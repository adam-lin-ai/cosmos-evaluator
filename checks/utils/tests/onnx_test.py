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

"""Unit tests for onnx module."""

import builtins
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import onnxruntime as ort

from checks.utils.onnx import clear_gpu_memory, configure_onnx_logging, get_inference_session


class TestConfigureOnnxLogging(unittest.TestCase):
    """Test cases for configure_onnx_logging function."""

    @patch("onnxruntime.set_default_logger_severity")
    @patch("warnings.filterwarnings")
    def test_configure_onnx_logging_not_verbose(self, mock_filterwarnings, mock_set_severity):
        """Test ONNX logging configuration in non-verbose mode."""
        configure_onnx_logging(verbose=False)

        # Should set logger severity to 3 (error level)
        mock_set_severity.assert_called_once_with(3)

        # Should filter ONNX runtime warnings
        mock_filterwarnings.assert_called_once_with("ignore", category=UserWarning, module="onnxruntime")

    @patch("onnxruntime.set_default_logger_severity")
    @patch("warnings.filterwarnings")
    def test_configure_onnx_logging_verbose(self, mock_filterwarnings, mock_set_severity):
        """Test ONNX logging configuration in verbose mode."""
        configure_onnx_logging(verbose=True)

        # Should not set logger severity in verbose mode
        mock_set_severity.assert_not_called()

        # Should not filter warnings in verbose mode
        mock_filterwarnings.assert_not_called()

    @patch("onnxruntime.set_default_logger_severity")
    @patch("warnings.filterwarnings")
    def test_configure_onnx_logging_default(self, mock_filterwarnings, mock_set_severity):
        """Test ONNX logging configuration with default parameters."""
        configure_onnx_logging()  # Default is verbose=False

        # Should behave like non-verbose mode
        mock_set_severity.assert_called_once_with(3)
        mock_filterwarnings.assert_called_once()


class TestClearGpuMemory(unittest.TestCase):
    """Test cases for clear_gpu_memory function."""

    @patch("gc.collect")
    def test_clear_gpu_memory_without_torch(self, mock_gc_collect):
        """Test clear_gpu_memory when torch is not available."""
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            # Should not raise even if torch import fails
            clear_gpu_memory()
            mock_gc_collect.assert_called_once()

    @patch("gc.collect")
    def test_clear_gpu_memory_with_torch_cuda_available(self, mock_gc_collect):
        """Test clear_gpu_memory when torch CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("checks.utils.onnx.logger"):
                # Re-import to get the patched version
                import importlib

                import checks.utils.onnx

                importlib.reload(checks.utils.onnx)

                checks.utils.onnx.clear_gpu_memory()

                # Verify gc.collect was called
                mock_gc_collect.assert_called()

                # Verify cuda.empty_cache and cuda.synchronize were called
                mock_torch.cuda.empty_cache.assert_called_once()
                mock_torch.cuda.synchronize.assert_called_once()

    @patch("gc.collect")
    def test_clear_gpu_memory_with_torch_cuda_not_available(self, mock_gc_collect):
        """Test clear_gpu_memory when torch CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("checks.utils.onnx.logger"):
                import importlib

                import checks.utils.onnx

                importlib.reload(checks.utils.onnx)

                checks.utils.onnx.clear_gpu_memory()

                mock_gc_collect.assert_called()

                # Verify cuda.empty_cache and cuda.synchronize were NOT called
                mock_torch.cuda.empty_cache.assert_not_called()
                mock_torch.cuda.synchronize.assert_not_called()


class TestGetInferenceSession(unittest.TestCase):
    """Test cases for get_inference_session function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = f"{self.temp_dir}/test_model.onnx"

        # Create a dummy ONNX model file
        with open(self.model_path, "wb") as f:
            f.write(b"dummy onnx model content")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("onnxruntime.InferenceSession")
    def test_get_inference_session_success(self, mock_inference_session):
        """Test successful inference session creation."""
        # Mock the inference session
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_inference_session.return_value = mock_session

        with patch("onnxruntime.get_device", return_value="GPU"):
            result = get_inference_session(self.model_path, verbose=False)

        # Should return the mocked session
        self.assertEqual(result, mock_session)

        # Verify InferenceSession was called with correct parameters
        self.assertEqual(mock_inference_session.call_count, 1)
        call_args = mock_inference_session.call_args

        # Check model path
        self.assertEqual(call_args[0][0], self.model_path)

        # Check providers
        providers = call_args[1]["providers"]
        self.assertIn(
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 16 * 1024 * 1024 * 1024,  # 16GB
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv1d_pad_to_nc1d": True,
                    "do_copy_in_default_stream": True,
                    "enable_cuda_graph": False,
                    "cudnn_conv_use_max_workspace": False,
                },
            ),
            providers,
        )
        self.assertIn("CPUExecutionProvider", providers)

        # Check session options
        sess_options = call_args[1]["sess_options"]
        self.assertEqual(sess_options.log_severity_level, 2)  # Not verbose
        self.assertTrue(sess_options.enable_cpu_mem_arena)
        self.assertTrue(sess_options.enable_mem_pattern)
        self.assertEqual(sess_options.intra_op_num_threads, 0)  # Auto-detect all cores
        self.assertEqual(sess_options.graph_optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        self.assertEqual(sess_options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertFalse(sess_options.enable_profiling)

    @patch("onnxruntime.InferenceSession")
    def test_get_inference_session_verbose(self, mock_inference_session):
        """Test inference session creation in verbose mode."""
        # Mock the inference session and its methods
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider"]

        # Mock inputs and outputs
        mock_input = MagicMock()
        mock_input.name = "input_tensor"
        mock_input.shape = [1, 3, 224, 224]
        mock_session.get_inputs.return_value = [mock_input]

        mock_output = MagicMock()
        mock_output.name = "output_tensor"
        mock_output.shape = [1, 1000]
        mock_session.get_outputs.return_value = [mock_output]

        mock_inference_session.return_value = mock_session

        with patch("onnxruntime.get_device", return_value="GPU"):
            result = get_inference_session(self.model_path, verbose=True)

        # Should return the mocked session
        self.assertEqual(result, mock_session)

        # Check that verbose logging settings were applied
        call_args = mock_inference_session.call_args
        sess_options = call_args[1]["sess_options"]
        self.assertEqual(sess_options.log_severity_level, 0)  # Verbose

    @patch("onnxruntime.InferenceSession")
    def test_get_inference_session_fallback_to_cpu(self, mock_inference_session):
        """Test fallback to CPU when CUDA session creation fails."""
        # First call raises exception, second call succeeds
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        mock_inference_session.side_effect = [Exception("CUDA failed"), mock_session]

        result = get_inference_session(self.model_path, verbose=False)

        # Should return the CPU session
        self.assertEqual(result, mock_session)

        # Should have been called twice (CUDA failed, then CPU)
        self.assertEqual(mock_inference_session.call_count, 2)

        # Second call should use CPU provider only
        second_call_args = mock_inference_session.call_args_list[1]
        providers = second_call_args[1]["providers"]
        self.assertEqual(providers, ["CPUExecutionProvider"])

    @patch("onnxruntime.InferenceSession")
    @patch("onnxruntime.SessionOptions")
    def test_session_options_configuration(self, mock_session_options, mock_inference_session):
        """Test that session options are configured correctly."""
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_inference_session.return_value = mock_session

        mock_options = MagicMock()
        mock_session_options.return_value = mock_options

        get_inference_session(self.model_path, verbose=False)

        # Verify session options were configured
        self.assertEqual(mock_options.log_severity_level, 2)
        self.assertTrue(mock_options.enable_cpu_mem_arena)
        self.assertTrue(mock_options.enable_mem_pattern)
        self.assertEqual(mock_options.intra_op_num_threads, 0)
        self.assertEqual(mock_options.graph_optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    @patch("onnxruntime.InferenceSession")
    def test_provider_configuration(self, mock_inference_session):
        """Test that providers are configured correctly."""
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_inference_session.return_value = mock_session

        get_inference_session(self.model_path)

        # Verify providers were configured correctly
        call_args = mock_inference_session.call_args
        providers = call_args[1]["providers"]

        # Should have CUDA provider with specific settings
        cuda_provider = providers[0]
        self.assertEqual(cuda_provider[0], "CUDAExecutionProvider")
        self.assertEqual(cuda_provider[1]["cudnn_conv_algo_search"], "HEURISTIC")

        # Should have CPU provider as fallback
        self.assertEqual(providers[1], "CPUExecutionProvider")

    @patch("onnxruntime.InferenceSession")
    def test_model_info_logging_verbose(self, mock_inference_session):
        """Test that model info is logged in verbose mode."""
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider"]

        # Mock model inputs
        mock_input1 = MagicMock()
        mock_input1.name = "input1"
        mock_input1.shape = [1, 3, 224, 224]

        mock_input2 = MagicMock()
        mock_input2.name = "input2"
        mock_input2.shape = [1, 100]

        mock_session.get_inputs.return_value = [mock_input1, mock_input2]

        # Mock model outputs
        mock_output1 = MagicMock()
        mock_output1.name = "output1"
        mock_output1.shape = [1, 1000]

        mock_session.get_outputs.return_value = [mock_output1]

        mock_inference_session.return_value = mock_session

        with patch("onnxruntime.get_device", return_value="GPU"):
            result = get_inference_session(self.model_path, verbose=True)

        # Verify that input and output information was queried
        mock_session.get_inputs.assert_called_once()
        mock_session.get_outputs.assert_called_once()

        self.assertEqual(result, mock_session)

    def test_nonexistent_model_path(self):
        """Test with non-existent model path."""
        nonexistent_path = "/path/to/nonexistent/model.onnx"

        # This should raise an exception when trying to create the session
        with self.assertRaises(Exception):
            get_inference_session(nonexistent_path)


if __name__ == "__main__":
    unittest.main()
