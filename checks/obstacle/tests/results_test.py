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

import os
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from checks.obstacle.results import get_object_type_track_idxs, load_results_from_json, save_results_to_json


class TestResultsIO(unittest.TestCase):
    nan_value = (
        -999.0
    )  # the actual value doesn't really matter for testing purposes; it is just to be able to use assert_allclose

    def _make_results(self, shape=(3, 4)):
        rng = np.random.default_rng(42)
        mat = rng.random(shape)
        # sprinkle NaNs
        mat[0, 1] = np.nan
        mat[-1, -2] = np.nan
        results = {
            "track_ids": [3, 1, 2],
            "processed_frame_ids": [10, 11, 12],
            "num_tracks": 3,
            "score_matrix": mat,
        }
        return results

    def _roundtrip(self, results, matrix_format, output_file_prefix=None):
        with tempfile.TemporaryDirectory() as td:
            path = save_results_to_json(
                results,
                clip_id="clip123",
                output_dir=td,
                matrix_format=matrix_format,
                output_file_prefix=output_file_prefix,
            )
            self.assertTrue(os.path.exists(path))

            # basic JSON sanity
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                self.assertIn('"metadata"', text)
                self.assertIn(f'"{matrix_format}"', text)

            loaded = load_results_from_json(path)
            return path, loaded

    def test_sparse_roundtrip(self):
        res = self._make_results()
        path, loaded = self._roundtrip(res, "sparse", output_file_prefix="sparse")

        # check non-matrix fields
        self.assertEqual(sorted(loaded["track_ids"]), [1, 2, 3])
        self.assertEqual(loaded["processed_frame_ids"], [10, 11, 12])
        self.assertEqual(loaded["metadata"]["matrix_format"], "sparse")

        # check matrix with NaNs
        assert_allclose(
            np.nan_to_num(loaded["score_matrix"], nan=self.nan_value),
            np.nan_to_num(res["score_matrix"], nan=self.nan_value),
        )
        self.assertTrue(np.isnan(loaded["score_matrix"][0, 1]))
        self.assertTrue(np.isnan(loaded["score_matrix"][-1, -2]))

        # ensure file suffix correct
        self.assertTrue(path.endswith("clip123.sparse.object.results.json"))

    def test_dense_roundtrip(self):
        res = self._make_results()
        _, loaded = self._roundtrip(res, "dense")
        assert_allclose(
            np.nan_to_num(loaded["score_matrix"], nan=self.nan_value),
            np.nan_to_num(res["score_matrix"], nan=self.nan_value),
        )
        self.assertTrue(np.isnan(loaded["score_matrix"][0, 1]))

    def test_compressed_roundtrip(self):
        res = self._make_results()
        _, loaded = self._roundtrip(res, "compressed")
        # dtype will be float64 due to tobytes/frombuffer path
        self.assertEqual(loaded["score_matrix"].dtype, np.float64)
        assert_allclose(
            np.nan_to_num(loaded["score_matrix"], nan=self.nan_value),
            np.nan_to_num(res["score_matrix"], nan=self.nan_value),
        )
        self.assertTrue(np.isnan(loaded["score_matrix"][0, 1]))

    def test_summary_saves_no_matrix(self):
        res = self._make_results()
        path, loaded = self._roundtrip(res, "summary")
        self.assertIsNone(loaded["score_matrix"])  # cannot reconstruct

    def test_compact_array_formatting_in_dense(self):
        # craft a tiny dense matrix with simple integers to make formatting predictable
        res = {
            "track_ids": {1},
            "processed_frame_ids": [0, 1],
            "num_tracks": 1,
            "score_matrix": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        with tempfile.TemporaryDirectory() as td:
            path = save_results_to_json(res, clip_id="abc", output_dir=td, matrix_format="dense")
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
                # ensure arrays are compact i.e., no newlines in the small numeric arrays
                self.assertIn("[1.0,2.0]", txt.replace(" ", ""))
                self.assertIn("[3.0,4.0]", txt.replace(" ", ""))


class TestGetObjectTypeTrackIdxs(unittest.TestCase):
    def test_basic_mapping(self):
        # track_ids must be an ordered sequence to define stable indices
        results = {
            "track_ids": [101, 102, 103, 104],
            "tracks": [
                {"track_id": 101, "object_type": "vehicle"},
                {"track_id": 102, "object_type": "pedestrian"},
                {"track_id": 104, "object_type": "vehicle"},
            ],
        }
        mapping = get_object_type_track_idxs(results)
        self.assertEqual(
            mapping,
            {
                "vehicle": [0, 3],
                "pedestrian": [1],
            },
        )

    def test_missing_keys_returns_empty(self):
        self.assertEqual(get_object_type_track_idxs({}), {})
        self.assertEqual(get_object_type_track_idxs({"track_ids": [1, 2]}), {})
        self.assertEqual(get_object_type_track_idxs({"tracks": []}), {})


class TestObjectTypeIndexSerialization(unittest.TestCase):
    """Test that object_type_index is properly serialized for static objects."""

    def test_object_type_index_roundtrip(self):
        """Verify object_type_index is preserved through save/load cycle."""
        results = {
            "track_ids": [1, 100, 101],
            "processed_frame_ids": [0, 1, 2],
            "score_matrix": np.array([[0.5, 0.8, 0.9], [0.6, 0.85, 0.95], [0.7, 0.9, 1.0]]),
            "tracks": [
                # Dynamic object (no object_type_index)
                {"track_id": 1, "object_type": "Car", "processor_output": [3, 0, 0]},
                # Static objects (with object_type_index)
                {"track_id": 100, "object_type": "LaneLine", "object_type_index": 19, "processor_output": [3, 0, 0]},
                {"track_id": 101, "object_type": "TrafficLight", "object_type_index": 8, "processor_output": [3, 0, 0]},
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            path = save_results_to_json(results, clip_id="test_clip", output_dir=td, matrix_format="sparse")
            loaded = load_results_from_json(path)

            # Verify tracks are loaded correctly
            self.assertEqual(len(loaded["tracks"]), 3)

            # Find tracks by track_id
            track_1 = next(t for t in loaded["tracks"] if t["track_id"] == 1)
            track_100 = next(t for t in loaded["tracks"] if t["track_id"] == 100)
            track_101 = next(t for t in loaded["tracks"] if t["track_id"] == 101)

            # Dynamic object should not have object_type_index
            self.assertNotIn("object_type_index", track_1)
            self.assertEqual(track_1["object_type"], "Car")

            # Static objects should have object_type_index preserved
            self.assertEqual(track_100["object_type_index"], 19)
            self.assertEqual(track_100["object_type"], "LaneLine")

            self.assertEqual(track_101["object_type_index"], 8)
            self.assertEqual(track_101["object_type"], "TrafficLight")


if __name__ == "__main__":
    unittest.main()
