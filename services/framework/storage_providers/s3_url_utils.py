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

"""Pure utility functions and data classes for S3 URL parsing and validation.

These utilities have no S3 client dependency — they operate purely on URL strings.
"""

from dataclasses import dataclass
import re
from urllib.parse import parse_qs, unquote, urlparse


@dataclass
class S3Urls:
    """Data class for S3 URLs."""

    raw: str | None = None
    presigned: str | None = None


@dataclass
class S3UrlComponents:
    """Data class for parsed S3 URL components."""

    bucket_name: str
    s3_key: str
    region_name: str = "us-east-1"  # Default region


def parse_s3_url(url: str) -> S3UrlComponents:
    """Parse an S3 URL (presigned or regular) to extract bucket, key, and region.

    Supports the following URL formats:
    - Virtual-hosted style: https://{bucket}.s3.amazonaws.com/{key}
    - Virtual-hosted with region: https://{bucket}.s3.{region}.amazonaws.com/{key}
    - Path style: https://s3.{region}.amazonaws.com/{bucket}/{key}
    - S3 URI: s3://{bucket}/{key}

    For presigned URLs, the region is extracted from the X-Amz-Credential query parameter.

    Args:
        url: S3 URL to parse (can be presigned or regular)

    Returns:
        S3UrlComponents with bucket_name, s3_key, and region_name

    Raises:
        ValueError: If the URL format is not recognized as a valid S3 URL
    """
    parsed = urlparse(url)

    # Handle s3:// URI scheme
    if parsed.scheme == "s3":
        bucket_name = parsed.netloc
        s3_key = unquote(parsed.path.lstrip("/"))
        return S3UrlComponents(bucket_name=bucket_name, s3_key=s3_key)

    # Handle HTTPS URLs
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    hostname = parsed.hostname or ""
    path = unquote(parsed.path.lstrip("/"))
    region_name = "us-east-1"  # Default region

    # Try to extract region from X-Amz-Credential in presigned URL query params
    if parsed.query:
        # X-Amz-Credential format: {access_key}/{date}/{region}/s3/aws4_request
        credential_match = re.search(r"X-Amz-Credential=[^/]+/\d+/([^/]+)/s3/", unquote(parsed.query))
        if credential_match:
            region_name = credential_match.group(1)

    # Virtual-hosted style with region: {bucket}.s3.{region}.amazonaws.com
    match = re.match(r"^(.+)\.s3\.([a-z0-9-]+)\.amazonaws\.com$", hostname)
    if match:
        bucket_name = match.group(1)
        region_name = match.group(2)
        return S3UrlComponents(bucket_name=bucket_name, s3_key=path, region_name=region_name)

    # Virtual-hosted style without region: {bucket}.s3.amazonaws.com
    match = re.match(r"^(.+)\.s3\.amazonaws\.com$", hostname)
    if match:
        bucket_name = match.group(1)
        return S3UrlComponents(bucket_name=bucket_name, s3_key=path, region_name=region_name)

    # Path style: s3.{region}.amazonaws.com/{bucket}/{key}
    match = re.match(r"^s3\.([a-z0-9-]+)\.amazonaws\.com$", hostname)
    if match:
        region_name = match.group(1)
        path_parts = path.split("/", 1)
        if len(path_parts) < 2:
            raise ValueError(f"Invalid path-style S3 URL, missing key: {url}")
        bucket_name = path_parts[0]
        s3_key = path_parts[1]
        return S3UrlComponents(bucket_name=bucket_name, s3_key=s3_key, region_name=region_name)

    # Path style without region: s3.amazonaws.com/{bucket}/{key}
    if hostname == "s3.amazonaws.com":
        path_parts = path.split("/", 1)
        if len(path_parts) < 2:
            raise ValueError(f"Invalid path-style S3 URL, missing key: {url}")
        bucket_name = path_parts[0]
        s3_key = path_parts[1]
        return S3UrlComponents(bucket_name=bucket_name, s3_key=s3_key, region_name=region_name)

    raise ValueError(f"Unrecognized S3 URL format: {url}")


def is_presigned_url_s3(url: str) -> bool:
    """Determines if the given URL is an AWS S3 presigned URL.

    Checks for the presence of AWS Signature Version 4 query parameters
    that indicate a presigned URL, and validates that the hostname looks like S3.

    Args:
        url: URL to check

    Returns:
        True if the URL appears to be a presigned S3 URL, False otherwise
    """
    parsed = urlparse(url)

    if not parsed.query:
        return False

    # Validate hostname looks like S3 (virtual-hosted or path-style)
    hostname = parsed.hostname or ""
    is_s3_host = (
        # Virtual-hosted style: {bucket}.s3.amazonaws.com or {bucket}.s3.{region}.amazonaws.com
        (".s3." in hostname and hostname.endswith(".amazonaws.com"))
        # Path style: s3.amazonaws.com or s3.{region}.amazonaws.com
        or hostname == "s3.amazonaws.com"
        or (hostname.startswith("s3.") and hostname.endswith(".amazonaws.com"))
    )

    if not is_s3_host:
        return False

    # AWS Signature Version 4 required query parameters for presigned URLs
    required_params = {"X-Amz-Algorithm", "X-Amz-Credential", "X-Amz-Signature"}

    # Use parse_qs for robust query parameter parsing (handles encoding, empty values, etc.)
    query_params = parse_qs(parsed.query, keep_blank_values=True)

    return required_params.issubset(query_params.keys())
