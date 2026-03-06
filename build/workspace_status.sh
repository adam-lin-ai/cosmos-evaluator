#!/bin/bash

# Workspace status script for Bazel stamping
# Variables prefixed with STABLE_ go to stable-status.txt (triggers cache invalidation)
# Variables without STABLE_ prefix go to volatile-status.txt (frequently changing)

# STABLE VALUES - These will invalidate cache when changed
# ISO 8601 timestamp - putting this in stable status ensures cache invalidation
echo "STABLE_BUILD_ISO_TIMESTAMP $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Git commit SHA
echo "STABLE_GIT_SHA $(git rev-parse HEAD)"

# Build user (stable for a given environment)
echo "STABLE_BUILD_USER ${USER:-unknown}"

# Build host (stable for a given environment)
echo "STABLE_BUILD_HOST $(hostname)"

# VOLATILE VALUES - These change frequently but don't invalidate cache
# Current Unix timestamp (changes every second)
echo "BUILD_TIMESTAMP $(date +%s)"

# Human-readable formatted date
echo "FORMATTED_DATE $(date '+%Y %b %d %H %M %S %a')"
