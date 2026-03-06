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

"""Unit tests for multistorage utilities."""

import json
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from checks.utils.multistorage import download_if_remote, is_remote_path, setup_msc_config, validate_uri


class _FakeTempFile:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TestMultiStorage(unittest.TestCase):
    @patch("checks.utils.multistorage.logger.warning")
    def test_setup_msc_config_skips_when_missing(self, mock_warning: MagicMock) -> None:
        with patch.dict(os.environ, {}, clear=True):
            setup_msc_config(None)
            setup_msc_config("")

            self.assertNotIn("MSC_CONFIG", os.environ)
            self.assertEqual(mock_warning.call_count, 2)

    @patch("checks.utils.multistorage.os.close")
    @patch("checks.utils.multistorage.tempfile.mkstemp")
    @patch("checks.utils.multistorage.json.dump")
    def test_setup_msc_config_writes_and_sets_env(
        self, mock_json_dump: MagicMock, mock_mkstemp: MagicMock, mock_close: MagicMock
    ) -> None:
        payload = {"path_mapping": [{"name": "x"}]}
        config = json.dumps(payload)
        mock_mkstemp.return_value = (123, "/tmp/msc_config_test.json")

        with patch("builtins.open", mock_open()) as mocked_open:
            with patch.dict(os.environ, {}, clear=True):
                setup_msc_config(config)

                mock_mkstemp.assert_called_once()
                mock_close.assert_called_once_with(123)
                mocked_open.assert_called_once_with("/tmp/msc_config_test.json", "w")
                mock_json_dump.assert_called_once()
                self.assertEqual(os.environ["MSC_CONFIG"], "/tmp/msc_config_test.json")

    def test_is_remote_path(self) -> None:
        self.assertTrue(is_remote_path("s3://bucket/key"))
        self.assertFalse(is_remote_path("http://example.com/file.mp4"))
        self.assertTrue(is_remote_path("https://example.com/file.mp4"))
        self.assertFalse(is_remote_path("/tmp/local.mp4"))

    @patch("checks.utils.multistorage.msc.open")
    def test_download_if_remote_local_path(self, mock_msc_open: MagicMock) -> None:
        path = "/tmp/local.mp4"
        self.assertEqual(download_if_remote(path), path)
        mock_msc_open.assert_not_called()

    @patch("checks.utils.multistorage.tempfile.NamedTemporaryFile")
    @patch("checks.utils.multistorage.msc.open")
    def test_download_if_remote_remote_path(
        self,
        mock_msc_open: MagicMock,
        mock_named_tempfile: MagicMock,
    ) -> None:
        fake_temp = _FakeTempFile("/tmp/fake_download.mp4")
        mock_named_tempfile.return_value = fake_temp

        src = MagicMock()
        src.read.side_effect = [b"video-bytes", b""]
        mock_msc_open.return_value.__enter__.return_value = src

        with patch("builtins.open", mock_open()) as mocked_open:
            output = download_if_remote("s3://bucket/sample.mp4")

        self.assertEqual(output, "/tmp/fake_download.mp4")
        self.assertTrue(fake_temp.closed)
        mock_named_tempfile.assert_called_once_with(delete=False, suffix=".mp4")
        mock_msc_open.assert_called_once_with("s3://bucket/sample.mp4", "rb")
        mocked_open.assert_called_once_with("/tmp/fake_download.mp4", "wb")

    @patch("checks.utils.multistorage.msc.is_empty")
    @patch("checks.utils.multistorage.msc.is_file")
    def test_validate_uri_modes(self, mock_is_file: MagicMock, mock_is_empty: MagicMock) -> None:
        mock_is_file.return_value = True
        mock_is_empty.return_value = False

        self.assertTrue(validate_uri("s3://bucket/file.mp4", is_file=True))
        self.assertTrue(validate_uri("s3://bucket/dir", is_file=False))
        mock_is_file.assert_called_once_with("s3://bucket/file.mp4")
        mock_is_empty.assert_called_once_with("s3://bucket/dir")

    @patch("checks.utils.multistorage.msc.is_file")
    def test_validate_uri_exception(self, mock_is_file: MagicMock) -> None:
        mock_is_file.side_effect = RuntimeError("boom")
        self.assertFalse(validate_uri("s3://bucket/file.mp4", is_file=True))


if __name__ == "__main__":
    unittest.main()
