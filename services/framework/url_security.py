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

"""URL security utilities for SSRF prevention.

Provides helpers that reject URLs targeting private, reserved, loopback,
or link-local IP addresses.  Used both at request-validation time (early
feedback) and immediately before an HTTP fetch (TOCTOU / DNS-rebinding
mitigation).
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class InsecureUrlError(ValueError):
    """A URL targets a private/reserved address or uses an insecure scheme."""


def is_private_or_reserved_ip(ip_str: str) -> bool:
    """Return True if *ip_str* is a private, reserved, loopback, or link-local IP."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local


def hostname_resolves_to_private_ip(hostname: str) -> bool:
    """Return True if *hostname* resolves to any private/reserved IP address.

    Applies a 2-second socket timeout to avoid unbounded blocking on DNS
    lookups.  Returns False when the name cannot be resolved or the
    lookup times out.
    """
    original_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(2.0)
        results = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, socket.timeout):
        return False
    finally:
        socket.setdefaulttimeout(original_timeout)
    return any(is_private_or_reserved_ip(str(sockaddr[0])) for _, _, _, _, sockaddr in results)


def check_url_security(url: str) -> None:
    """Raise :class:`InsecureUrlError` if *url* targets a private/reserved address.

    Designed to be called immediately before an HTTP fetch so that
    DNS-rebinding attacks that pass an earlier Pydantic validation are
    still caught.

    Only ``https://`` URLs are checked; ``s3://``, ``gs://``, and local
    paths are left to their respective providers.
    """
    parsed = urlparse(url)

    if parsed.scheme == "http":
        raise InsecureUrlError(f"Unencrypted HTTP URLs are not allowed: {url}")

    if parsed.scheme == "https":
        hostname = parsed.hostname
        if not hostname:
            raise InsecureUrlError("HTTPS URL must include a hostname")

        if is_private_or_reserved_ip(hostname):
            raise InsecureUrlError(f"URL targets a private/reserved IP address: {hostname}")

        if hostname_resolves_to_private_ip(hostname):
            raise InsecureUrlError(f"URL hostname resolves to a private/reserved IP address: {hostname}")
