#!/usr/bin/env bash
# Builds FFmpeg from source with LGPL licensing, NVENC/CUVID hardware acceleration,
# and OpenH264 CPU-based H.264 encoding.
#
# Produces a tarball structured as an OCI image layer (files at usr/local/...)
# stored in Git LFS alongside this script.
#
# Usage:
#   ./third_party/ffmpeg/tar.sh
#
# After building, commit the tarball (tracked by Git LFS via .gitattributes).
#
# Requirements:
#   - Docker with access to nvcr.io/nvidia/cuda images

set -euo pipefail

FFMPEG_VERSION="${FFMPEG_VERSION:-8.0.1}"
NV_CODEC_HEADERS_VERSION="${NV_CODEC_HEADERS_VERSION:-n13.0.19.0}"
OPENH264_VERSION="${OPENH264_VERSION:-2.6.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="ffmpeg-${FFMPEG_VERSION}-lgpl-nvenc-openh264-linux-amd64.tar.gz"
OUTPUT="${SCRIPT_DIR}/${TARBALL}"

echo "==> Building FFmpeg ${FFMPEG_VERSION} with NVENC/CUVID + OpenH264 (LGPL)..."
docker run --rm -v "${SCRIPT_DIR}:/output" \
  nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 bash -c "
    set -euxo pipefail

    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake curl git meson nasm pkg-config xz-utils

    # nv-codec-headers (MIT-licensed) for NVENC/CUVID support
    git clone --depth 1 --branch ${NV_CODEC_HEADERS_VERSION} \
      https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers
    make -C /tmp/nv-codec-headers -j\$(nproc)
    make -C /tmp/nv-codec-headers install

    # OpenH264 (BSD-2-Clause) for CPU-based H.264 encoding
    curl -fsSL https://github.com/cisco/openh264/archive/refs/tags/v${OPENH264_VERSION}.tar.gz \
      | tar -xz -C /tmp
    cd /tmp/openh264-${OPENH264_VERSION}
    meson setup builddir --prefix=/usr/local --default-library=shared --buildtype=release
    meson compile -C builddir
    meson install -C builddir
    ldconfig

    # Download and build FFmpeg
    curl -fsSL https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz | tar -xJ -C /tmp
    cd /tmp/ffmpeg-${FFMPEG_VERSION}

    ./configure \
      --prefix=/opt/ffmpeg \
      --disable-debug \
      --disable-doc \
      --disable-static \
      --enable-shared \
      --enable-pic \
      --disable-nonfree \
      --disable-gpl \
      --disable-vulkan \
      --disable-cuda-nvcc \
      --enable-cuvid \
      --enable-nvenc \
      --enable-libopenh264

    make -j\$(nproc)
    make install

    # Bundle the OpenH264 shared library alongside FFmpeg's own libs
    cp /usr/local/lib/x86_64-linux-gnu/libopenh264.so* /opt/ffmpeg/lib/ 2>/dev/null \
      || cp /usr/local/lib/libopenh264.so* /opt/ffmpeg/lib/

    # Package runtime files as an OCI layer tar (paths prefixed with usr/local/)
    cd /opt/ffmpeg
    tar -czf /output/${TARBALL} --transform='s,^,usr/local/,' \
      bin/ffmpeg bin/ffprobe \
      \$(find lib -maxdepth 1 -name '*.so*' \\( -type f -o -type l \\))
  "

echo "==> Built: ${OUTPUT}"
sha256sum "${OUTPUT}"
echo "==> Commit this file to Git (tracked by LFS via .gitattributes)."
