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

"""Merge overlay dpkg stanzas into the base OCI image's /var/lib/dpkg/status.

Reads the base image's OCI layout to extract /var/lib/dpkg/status, then merges
overlay package stanzas from a rules_distroless dpkg_status tar. Packages
present in both are replaced with the overlay version (upgraded). The merged
result is written as a tar containing /var/lib/dpkg/status.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import sys
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

_DPKG_STATUS_PATH = "./var/lib/dpkg/status"
_DPKG_STATUS_PATH_NO_PREFIX = "var/lib/dpkg/status"
_DPKG_INFO_DIR = "./var/lib/dpkg/info"


def _parse_stanzas(text: str) -> dict[str, str]:
    """Parse dpkg status text into {package_name: stanza_text} dict."""
    stanzas: dict[str, str] = {}
    for stanza in text.split("\n\n"):
        stanza = stanza.strip()
        if not stanza:
            continue
        for line in stanza.splitlines():
            if line.startswith("Package:"):
                pkg_name = line.split(":", 1)[1].strip()
                stanzas[pkg_name] = stanza
                break
    return stanzas


def _get_field(stanza: str, field: str) -> str | None:
    """Extract a field value from a dpkg stanza."""
    for line in stanza.splitlines():
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return None


def _dpkg_info_name(stanza: str, pkg_name: str) -> str:
    """Return the dpkg info filename stem (e.g. 'libssl3:amd64' or 'gnupg2')."""
    if _get_field(stanza, "Multi-Arch") == "same":
        arch = _get_field(stanza, "Architecture") or "amd64"
        return f"{pkg_name}:{arch}"
    return pkg_name


def _extract_base_dpkg_status(oci_layout_dir: Path) -> str:
    """Extract /var/lib/dpkg/status from an OCI image layout.

    Walks layers bottom-to-top so the last layer that contains the file wins
    (matching OCI overlay semantics).
    """
    index = json.loads((oci_layout_dir / "index.json").read_text())
    manifest_digest = index["manifests"][0]["digest"]
    manifest_blob = oci_layout_dir / "blobs" / manifest_digest.replace(":", "/")
    manifest = json.loads(manifest_blob.read_text())

    status_text = ""
    for layer_desc in manifest["layers"]:
        layer_digest = layer_desc["digest"]
        layer_path = oci_layout_dir / "blobs" / layer_digest.replace(":", "/")
        media_type = layer_desc.get("mediaType", "")

        try:
            opener = gzip.open if ("gzip" in media_type or layer_path.suffix == ".gz") else open
            with opener(layer_path, "rb") as raw, tarfile.open(fileobj=raw, mode="r|*") as tf:
                for member in tf:
                    if member.name in (_DPKG_STATUS_PATH_NO_PREFIX, _DPKG_STATUS_PATH):
                        extracted = tf.extractfile(member)
                        if extracted:
                            status_text = extracted.read().decode("utf-8")
                        break
        except (tarfile.TarError, gzip.BadGzipFile, OSError):
            logger.warning("Skipping layer %s: unable to read as tar archive", layer_path, exc_info=True)
            continue

    return status_text


def _extract_overlay_stanzas(dpkg_status_tar: Path) -> str:
    """Extract /var/lib/dpkg/status text from a rules_distroless dpkg_status tar."""
    with tarfile.open(dpkg_status_tar, "r:*") as tf:
        try:
            member = tf.getmember(_DPKG_STATUS_PATH)
        except KeyError:
            msg = f"_extract_overlay_stanzas: member {_DPKG_STATUS_PATH!r} not found in {dpkg_status_tar}"
            raise ValueError(msg) from None
        f = tf.extractfile(member)
        if f is None:
            msg = f"_extract_overlay_stanzas: cannot read {_DPKG_STATUS_PATH!r} from {dpkg_status_tar} (not a regular file)"
            raise ValueError(msg)
        return f.read().decode("utf-8")


def main() -> None:
    """Merge base and overlay dpkg status files.

    Command-line arguments:
        argv[1]: Path to OCI layout directory containing the base image
        argv[2]: Path to overlay dpkg_status tar file
        argv[3]: Output path for merged dpkg_status tar file
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")

    if len(sys.argv) < 4:
        logger.error("Usage: %s <oci_layout_dir> <overlay_dpkg_status.tar> <output.tar>", sys.argv[0])
        sys.exit(1)

    oci_layout_dir = Path(sys.argv[1])
    overlay_tar = Path(sys.argv[2])
    output_tar = sys.argv[3]

    logger.info("OCI layout dir: %s", oci_layout_dir)
    logger.info("Overlay tar:    %s", overlay_tar)
    logger.info("Output tar:     %s", output_tar)

    if not oci_layout_dir.is_dir():
        logger.error("OCI layout directory not found: %s", oci_layout_dir)
        sys.exit(1)
    if not (oci_layout_dir / "index.json").is_file():
        logger.error("Not a valid OCI layout (missing index.json): %s", oci_layout_dir)
        sys.exit(1)
    if not overlay_tar.is_file():
        logger.error("Overlay dpkg_status tar not found: %s", overlay_tar)
        sys.exit(1)

    base_text = _extract_base_dpkg_status(oci_layout_dir)
    overlay_text = _extract_overlay_stanzas(overlay_tar)

    base_stanzas = _parse_stanzas(base_text)
    overlay_stanzas = _parse_stanzas(overlay_text)
    logger.info("Base packages: %d, overlay packages: %d", len(base_stanzas), len(overlay_stanzas))

    merged = {**base_stanzas, **overlay_stanzas}
    logger.info("Merged package count: %d", len(merged))

    merged_text = "\n\n".join(merged[pkg] for pkg in sorted(merged)) + "\n"
    merged_bytes = merged_text.encode("utf-8")

    try:
        with tarfile.open(output_tar, "w:") as tout:
            info = tarfile.TarInfo(name=_DPKG_STATUS_PATH)
            info.size = len(merged_bytes)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            tout.addfile(info, io.BytesIO(merged_bytes))
            logger.info("Wrote %s (%d bytes)", _DPKG_STATUS_PATH, len(merged_bytes))

            # Create empty .list files for overlay-only packages so dpkg -i doesn't
            # warn "files list file for package '...' missing". Skip packages that
            # also exist in the base image — those already have populated .list files.
            overlay_only = set(overlay_stanzas) - set(base_stanzas)
            for pkg_name in sorted(overlay_only):
                stanza = overlay_stanzas[pkg_name]
                list_name = _dpkg_info_name(stanza, pkg_name)
                list_path = f"{_DPKG_INFO_DIR}/{list_name}.list"
                list_info = tarfile.TarInfo(name=list_path)
                list_info.size = 0
                list_info.mode = 0o644
                list_info.uid = 0
                list_info.gid = 0
                tout.addfile(list_info)
                logger.debug("Created empty .list: %s", list_path)
            logger.info("Created %d empty .list files for overlay-only packages", len(overlay_only))
    except (tarfile.TarError, OSError):
        logger.exception("Failed to write output tar: %s", output_tar)
        sys.exit(1)


if __name__ == "__main__":
    main()
