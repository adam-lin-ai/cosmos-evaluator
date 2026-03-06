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

import logging
import sys
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from services.vlm.service import PresetRequest, PresetResponse, Request, Response, Service


class TestPresetRequestUrlSecurity:
    """Test SSRF protections on PresetRequest.augmented_video_url."""

    def _make_request(self, url: str) -> PresetRequest:
        return PresetRequest(
            augmented_video_url=url,
            preset_conditions={"weather": "sunny"},
            preset_check_config={"model": {"endpoint": "test"}},
        )

    def test_rejects_http_scheme(self):
        """Unencrypted HTTP must be rejected."""
        with pytest.raises(ValidationError, match="Unencrypted HTTP"):
            self._make_request("http://example.com/video.mp4")

    def test_allows_https_with_public_host(self):
        """HTTPS to a public hostname should be accepted."""
        with patch("services.vlm.service.hostname_resolves_to_private_ip", return_value=False):
            req = self._make_request("https://storage.example.com/video.mp4")
            assert req.augmented_video_url == "https://storage.example.com/video.mp4"

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
        ],
    )
    def test_rejects_private_ip_literals(self, ip: str):
        """HTTPS URLs with private/reserved IP literals must be rejected."""
        with pytest.raises(ValidationError, match="private/reserved IP"):
            self._make_request(f"https://{ip}/video.mp4")

    def test_rejects_loopback_ipv6(self):
        """HTTPS URL targeting IPv6 loopback must be rejected."""
        with pytest.raises(ValidationError, match="private/reserved IP"):
            self._make_request("https://[::1]/video.mp4")

    def test_rejects_hostname_resolving_to_private_ip(self):
        """HTTPS URL whose hostname resolves to a private IP must be rejected."""
        with patch("services.vlm.service.hostname_resolves_to_private_ip", return_value=True):
            with pytest.raises(ValidationError, match="resolves to a private/reserved IP"):
                self._make_request("https://internal.example.com/video.mp4")

    def test_allows_s3_url(self):
        """S3 URLs pass through — provider-specific validation is not the model's job."""
        req = self._make_request("s3://my-bucket/path/to/video.mp4")
        assert req.augmented_video_url == "s3://my-bucket/path/to/video.mp4"

    def test_allows_gs_url(self):
        """GCS URLs pass through — cloud-agnostic."""
        req = self._make_request("gs://my-bucket/path/to/video.mp4")
        assert req.augmented_video_url == "gs://my-bucket/path/to/video.mp4"

    def test_allows_local_path(self):
        """Local filesystem paths pass through unchanged."""
        req = self._make_request("/data/videos/test.mp4")
        assert req.augmented_video_url == "/data/videos/test.mp4"

    def test_https_without_hostname_rejected(self):
        """HTTPS URL with no hostname is rejected."""
        with pytest.raises(ValidationError, match="must include a hostname"):
            self._make_request("https:///path/only")


class TestPresetRequest:
    """Test cases for PresetRequest Pydantic model."""

    def test_valid_preset_request(self):
        """Test creating a valid PresetRequest."""
        request_data = {
            "augmented_video_url": "/path/to/video.mp4",
            "preset_conditions": {"weather": "sunny", "time_of_day": "day"},
            "preset_check_config": {"model": {"endpoint": "test_endpoint"}},
        }
        request = PresetRequest(**request_data)

        assert request.augmented_video_url == "/path/to/video.mp4"
        assert request.preset_conditions == {"weather": "sunny", "time_of_day": "day"}
        assert request.preset_check_config == {"model": {"endpoint": "test_endpoint"}}

    def test_missing_required_fields(self):
        """Test PresetRequest validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            PresetRequest()

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "augmented_video_url" in error_fields
        assert "preset_conditions" in error_fields

    def test_empty_augmented_video_url(self):
        """Test PresetRequest with empty augmented video URL."""
        with pytest.raises(ValidationError):
            PresetRequest(augmented_video_url="", preset_conditions={}, preset_check_config={})


class TestPresetResponse:
    """Test cases for PresetResponse Pydantic model."""

    def test_valid_preset_response(self):
        """Test creating a valid PresetResponse."""
        response_data = {
            "result": {
                "overall_score": 0.85,
                "scoring_details": {"weather_match": True},
                "frames_used": 12,
                "processing_time_s": 2.5,
                "model": "test_model",
            }
        }
        response = PresetResponse(**response_data)

        assert response.result == response_data["result"]

    def test_missing_result_field(self):
        """Test PresetResponse validation with missing result field."""
        with pytest.raises(ValidationError) as exc_info:
            PresetResponse()

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("result",)


class TestRequest:
    """Test cases for Request Pydantic model."""

    def test_valid_request(self):
        """Test creating a valid Request."""
        preset_request = PresetRequest(
            augmented_video_url="/path/to/video.mp4",
            preset_conditions={"weather": "sunny"},
            preset_check_config={"model": {"endpoint": "test"}},
        )
        request = Request(preset_request=preset_request)

        assert request.preset_request == preset_request

    def test_missing_preset_request(self):
        """Test Request creation with missing preset_request field (should be allowed with None default)."""
        request = Request()
        assert request.preset_request is None


class TestResponse:
    """Test cases for Response Pydantic model."""

    def test_valid_response(self):
        """Test creating a valid Response."""
        preset_response = PresetResponse(result={"test": "data"})
        response = Response(preset_response=preset_response)

        assert response.preset_response == preset_response

    def test_missing_preset_response(self):
        """Test Response validation with missing preset_response field."""
        with pytest.raises(ValidationError) as exc_info:
            Response()

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("preset_response",)


class TestService:
    """Test cases for the Service class."""

    @pytest.fixture
    def service(self):
        """Create a Service instance."""
        return Service()

    def test_service_initialization(self):
        """Test Service class initialization."""
        service = Service()

        assert service.logger is not None
        assert isinstance(service.logger, logging.Logger)
        assert service.logger.name == "services.vlm.service"

    @pytest.mark.asyncio
    async def test_process_raises_not_implemented(self, service):
        """process() is unused — the REST layer calls process_preset() directly."""
        request = Request(
            preset_request=PresetRequest(
                augmented_video_url="/path/to/video.mp4",
                preset_conditions={"weather": "sunny"},
                preset_check_config={"model": {"endpoint": "test"}},
            )
        )
        with pytest.raises(NotImplementedError):
            await service.process(request)

    @pytest.mark.asyncio
    async def test_get_default_config_success(self, service):
        """Test successful retrieval of default configuration."""
        expected_config = {
            "preset_check": {"model": {"endpoint": "default_endpoint"}, "keyframe_interval_s": 2.0, "max_frames": 24}
        }

        with patch("services.vlm.service.PresetProcessor.get_default_config") as mock_get_config:
            mock_get_config.return_value = expected_config

            result = await service.get_default_config()

            assert result == {"av.vlm": expected_config}
            mock_get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_default_config_exception(self, service):
        """Test handling of exceptions in get_default_config."""
        with patch("services.vlm.service.PresetProcessor.get_default_config") as mock_get_config:
            mock_get_config.side_effect = Exception("Config loading failed")

            with pytest.raises(Exception, match="Config loading failed"):
                await service.get_default_config()

    def test_service_inheritance(self, service):
        """Test that Service properly inherits from ServiceBase."""
        from services.framework.service_base import ServiceBase

        assert isinstance(service, ServiceBase)

    @pytest.mark.asyncio
    async def test_validate_input_success(self, service):
        """Test successful input validation."""
        preset_request = PresetRequest(
            augmented_video_url="/path/to/video.mp4",
            preset_conditions={"weather": "sunny"},
            preset_check_config={"model": {"endpoint": "test"}},
        )
        request = Request(preset_request=preset_request)

        is_valid = await service.validate_input(request)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_input_empty_augmented_video_url(self, service):
        """Test validation failure with whitespace-only PresetRequest.augmented_video_url."""
        preset_request = PresetRequest(
            augmented_video_url="   ",  # whitespace only
            preset_conditions={"weather": "sunny"},
            preset_check_config={"model": {"endpoint": "test"}},
        )
        request = Request(preset_request=preset_request)

        is_valid = await service.validate_input(request)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_input_empty_conditions(self, service):
        """Test validation failure with empty preset conditions."""
        preset_request = PresetRequest(
            augmented_video_url="/path/to/video.mp4",
            preset_conditions={},
            preset_check_config={"model": {"endpoint": "test"}},
        )
        request = Request(preset_request=preset_request)

        is_valid = await service.validate_input(request)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_input_empty_config(self, service):
        """Test validation success with empty preset check config."""
        preset_request = PresetRequest(
            augmented_video_url="/path/to/video.mp4", preset_conditions={"weather": "sunny"}, preset_check_config={}
        )
        request = Request(preset_request=preset_request)

        is_valid = await service.validate_input(request)
        assert is_valid is True

    def test_validate_input_preset_success(self):
        """Test successful validation of preset request."""
        preset_request = PresetRequest(
            augmented_video_url="/path/to/video.mp4",
            preset_conditions={"weather": "sunny"},
            preset_check_config={"model": {"endpoint": "test"}},
        )

        is_valid = Service.validate_input_preset(preset_request)
        assert is_valid is True

    def test_validate_input_preset_empty_augmented_video_url(self):
        """Test validation failure with empty augmented video URL."""
        with pytest.raises(ValidationError) as exc_info:
            PresetRequest(
                augmented_video_url="",
                preset_conditions={"weather": "sunny"},
                preset_check_config={"model": {"endpoint": "test"}},
            )

        errors = exc_info.value.errors()
        assert any(error["type"] == "string_too_short" for error in errors)
        assert any("augmented_video_url" in str(error["loc"]) for error in errors)

    def test_validate_input_preset_whitespace_augmented_video_url(self, service):
        """Test validation failure with whitespace-only augmented video URL."""
        preset_request = PresetRequest(
            augmented_video_url="   ",
            preset_conditions={"weather": "sunny"},
            preset_check_config={"model": {"endpoint": "test"}},
        )

        is_valid = service.validate_input_preset(preset_request)
        assert is_valid is False

    def test_validate_input_preset_empty_conditions(self, service):
        """Test validation failure with empty preset conditions."""
        preset_request = PresetRequest(
            augmented_video_url="/path/to/video.mp4",
            preset_conditions={},
            preset_check_config={"model": {"endpoint": "test"}},
        )

        is_valid = service.validate_input_preset(preset_request)
        assert is_valid is False

    def test_validate_input_preset_empty_config(self, service):
        """Test validation success with empty preset check config."""
        preset_request = PresetRequest(
            augmented_video_url="/path/to/video.mp4", preset_conditions={"weather": "sunny"}, preset_check_config={}
        )

        is_valid = service.validate_input_preset(preset_request)
        assert is_valid is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
