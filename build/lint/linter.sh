#!/usr/bin/env bash
#
# Linting workflow for Bazel targets.
#
# This script runs on the host and calls dazel commands to enter the container.
# See https://docs.aspect.build/cli/commands/aspect_lint
#
# By default, shows violations without applying patches. Use --apply-patches to fix violations.
#
# Usage:
#   linter.sh [--fail-on-violation] [--apply-patches] [--output-base=PATH] [TARGETS...]
#
# Compatible with both bash and zsh.
#

# Enable zsh compatibility options if running in zsh
if [[ -n "${ZSH_VERSION:-}" ]]; then
    setopt SH_WORD_SPLIT  # Enable word splitting like bash
    setopt KSH_ARRAYS     # Use 0-based array indexing like bash
fi

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default options
APPLY_PATCHES=false
BUILDIFIER=true
FAIL_ON_VIOLATION=false
MYPY=true
OUTPUT_BASE=""
TARGETS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --apply-patches)
            APPLY_PATCHES=true
            shift
            ;;
        --fail-on-violation)
            FAIL_ON_VIOLATION=true
            shift
            ;;
        --no-buildifier)
            BUILDIFIER=false
            shift
            ;;
        --no-mypy)
            MYPY=false
            shift
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --output-base=*)
            OUTPUT_BASE="${1#*=}"
            shift
            ;;
        --help|-h)
            cat << EOF
Usage: linter.sh [OPTIONS] [TARGETS...]

Run linters on Bazel targets

Options:
  --apply-patches           Auto-fix violations by applying patches (default: false)
  --fail-on-violation       Fail the build when a linter warning is present (default: false)
  --no-buildifier           Disable buildifier linting (default: false)
  --no-mypy                 Disable mypy type checking (default: false)
  --output-base=PATH        Set the output_base directory for dazel/bazel (default: none)

Arguments:
  TARGETS                   Bazel target patterns to lint (e.g., //... or //checks/...)
                            (default: //...)

Examples:
  linter.sh                              # Check for violations (default)
  linter.sh //checks/...                 # Check specific targets
  linter.sh --apply-patches              # Fix violations in everything
  linter.sh --apply-patches //checks/... # Fix violations in specific targets
  linter.sh --fail-on-violation          # Check and fail if violations exist
  linter.sh --no-mypy                    # Run only ruff and buildifier (skip mypy)
  linter.sh --no-buildifier              # Run only ruff and mypy (skip buildifier)
EOF
            exit 0
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
        *)
            TARGETS+=("$1")
            shift
            ;;
    esac
done

# Default to //... if no targets specified
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("//...")
fi

build_dazel_cmd_array() {
    # Build dazel command array with optional output_base
    # Sets DAZEL_CMD_RESULT array (portable across bash/zsh)
    DAZEL_CMD_RESULT=()  # Clear any previous content

    # Use dazel command if available (e.g., from sourced env_setup.sh in bash)
    # Otherwise fall back to calling dazel.py directly (needed for zsh or unsourced shells)
    if command -v dazel &>/dev/null; then
        DAZEL_CMD_RESULT+=("dazel")
    else
        # Get the path to dazel.py relative to this script
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
        DAZEL_CMD_RESULT+=("$script_dir/../dazel/dazel.py")
    fi

    if [[ -n "$OUTPUT_BASE" ]]; then
        DAZEL_CMD_RESULT+=("--output_base=$OUTPUT_BASE")
    fi
}

find_workspace_root() {
    # Find workspace root using dazel
    local workspace
    build_dazel_cmd_array
    local -a dazel_cmd=("${DAZEL_CMD_RESULT[@]}")
    if workspace="$("${dazel_cmd[@]}" info workspace 2>/dev/null)" && [[ -n "$workspace" ]]; then
        echo "$workspace"
        return
    fi

    # Try git if dazel info fails
    if git rev-parse --show-toplevel &>/dev/null; then
        echo "$(git rev-parse --show-toplevel)"
        return
    fi

    echo "Error: Could not find workspace root" >&2
    exit 1
}

WORKSPACE_ROOT="$(find_workspace_root)"

echo "Workspace: $WORKSPACE_ROOT"
echo ""

# Create temporary file for build events in workspace root
BUILD_EVENTS_FILE=$(mktemp --tmpdir="$WORKSPACE_ROOT" --suffix=.json)

# Build bazel command
BAZEL_ARGS=(
    "build"
    "--build_event_json_file=$BUILD_EVENTS_FILE"
    "--experimental_proto_descriptor_sets_include_source_info"
    "--keep_going"
    "--remote_download_toplevel"
)

ASPECTS=("//build/lint:linters.bzl%ruff_aspect")
OUTPUT_GROUPS=("rules_lint_human")
if [[ "$MYPY" == "true" ]]; then
    ASPECTS+=("//build/lint:linters.bzl%mypy_aspect")
    OUTPUT_GROUPS+=("+mypy")
fi

# Configure patch generation based on mode
if [[ "$APPLY_PATCHES" == "true" ]]; then
    # Generate and apply patches
    BAZEL_ARGS+=("--@aspect_rules_lint//lint:fix")
    OUTPUT_GROUPS+=("rules_lint_patch")
fi

# Join arrays with commas
ASPECTS_STR=$(IFS=, ; echo "${ASPECTS[*]}")
BAZEL_ARGS+=("--aspects=${ASPECTS_STR}")

OUTPUT_GROUPS_STR=$(IFS=, ; echo "${OUTPUT_GROUPS[*]}")
BAZEL_ARGS+=("--output_groups=${OUTPUT_GROUPS_STR}")

# Add targets
BAZEL_ARGS+=("${TARGETS[@]}")

# Build dazel command array
build_dazel_cmd_array
DAZEL_CMD=("${DAZEL_CMD_RESULT[@]}")

# Run bazel build
echo "Running: ${DAZEL_CMD[*]} ${BAZEL_ARGS[*]}"
echo ""

# Create temporary file for bazel output in workspace root
BAZEL_OUTPUT_FILE=$(mktemp --tmpdir="$WORKSPACE_ROOT" --suffix=.txt)
trap 'rm -f "$BUILD_EVENTS_FILE" "$BAZEL_OUTPUT_FILE"' EXIT

# Run bazel and capture output and exit code
set +e
"${DAZEL_CMD[@]}" "${BAZEL_ARGS[@]}" 2>&1 | tee "$BAZEL_OUTPUT_FILE"
BAZEL_EXIT_CODE=$?
set -e

if [[ "$BUILDIFIER" == "true" ]]; then
    set +e
    "${DAZEL_CMD[@]}" run //build/lint:buildifier.check 2>&1 | tee -a "$BAZEL_OUTPUT_FILE"
    set -e
    # Note: buildifier exits with non-zero when it finds violations, but we ignore the exit code
    # The violations will be counted and handled in the TOTAL_VIOLATIONS logic below
fi

# Show linting results
echo ""
echo "========================================================================"
echo "RUFF RESULTS"
echo "========================================================================"
echo ""

TOTAL_VIOLATIONS=0
# Process ruff violations
if ! command -v jq &> /dev/null; then
    # Check if jq is available
    echo "Warning: jq not found, cannot extract report files" >&2
    echo "Install jq to see detailed reports: apt-get install jq" >&2
    exit 1
else
    # Parse build events to find .out files
    REPORT_FILES=$(jq -r '
        select(.namedSetOfFiles.files != null) |
        .namedSetOfFiles.files[] |
        select(.name | endswith(".out")) |
        select(.name | contains("AspectRulesLint")) |
        if .pathPrefix then
            (.pathPrefix | join("/")) + "/" + .name
        else
            .name
        end
    ' "$BUILD_EVENTS_FILE" 2>/dev/null | sort -u || true)

    # Process .out files
    FILES_WITH_RUFF_VIOLATIONS=0
    RUFF_VIOLATIONS=0

    while IFS= read -r report_file; do
        [[ -z "$report_file" ]] && continue
        full_path="$WORKSPACE_ROOT/$report_file"
        if [[ -f "$full_path" && -s "$full_path" ]]; then
            content=$(cat "$full_path")

            # Skip if "All checks passed!"
            if echo "$content" | grep -q "All checks passed!"; then
                continue
            fi

            # Extract source file from report path
            source_file=$(echo "$report_file" | sed -E 's/\.AspectRulesLint.*\.out$/.py/' | sed -E 's|.*/bin/||')

            # Count violations (look for "Found N error")
            violation_count=$(echo "$content" | grep -oP 'Found \K\d+(?= errors?)' || echo "1")

            FILES_WITH_RUFF_VIOLATIONS=$((FILES_WITH_RUFF_VIOLATIONS + 1))
            RUFF_VIOLATIONS=$((RUFF_VIOLATIONS + violation_count))

            echo "$source_file: $violation_count violation(s)"

            # Always show violation content
            echo "$content"
            echo ""
        fi
    done <<< "$REPORT_FILES"

    if [[ $FILES_WITH_RUFF_VIOLATIONS -eq 0 ]]; then
        echo "✓ No ruff violations found!"
    else
        echo ""
        echo "Total: $FILES_WITH_RUFF_VIOLATIONS file(s) with $RUFF_VIOLATIONS violation(s)"
        echo ""
        # Add to total violations count
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + RUFF_VIOLATIONS))
    fi
fi

# Process mypy violations
FILES_WITH_MYPY_VIOLATIONS=0
MYPY_VIOLATIONS=0

if [[ "$MYPY" == "true" ]]; then
    echo ""
    echo "========================================================================"
    echo "MYPY RESULTS"
    echo "========================================================================"
    echo ""

    # Extract mypy errors from bazel output
    # Pattern: "filepath.py:line: error: message [error-code]"
    MYPY_ERRORS=$(grep -E '^[a-zA-Z0-9_/.-]+\.py:[0-9]+: error:' "$BAZEL_OUTPUT_FILE" || true)

    if [[ -z "$MYPY_ERRORS" ]]; then
        echo "✓ No mypy violations found!"
    else
        # Count unique files with errors
        FILES_WITH_MYPY_VIOLATIONS=$(echo "$MYPY_ERRORS" | cut -d: -f1 | sort -u | wc -l)

        # Count total errors
        MYPY_VIOLATIONS=$(echo "$MYPY_ERRORS" | wc -l)

        # Display errors grouped by file
        while IFS= read -r file; do
            [[ -z "$file" ]] && continue

            file_errors=$(echo "$MYPY_ERRORS" | grep "^$file:")
            error_count=$(echo "$file_errors" | wc -l)

            echo "$file: $error_count error(s)"
            echo "$file_errors"
            echo ""
        done < <(echo "$MYPY_ERRORS" | cut -d: -f1 | sort -u)

        echo "Total: $FILES_WITH_MYPY_VIOLATIONS file(s) with $MYPY_VIOLATIONS error(s)"
        echo ""

        # Add to total violations count
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + MYPY_VIOLATIONS))
    fi
fi

# Process buildifier violations
BUILDIFIER_VIOLATION_COUNT=0
FILES_WITH_BUILDIFIER_VIOLATIONS=0
if [[ "$BUILDIFIER" == "true" ]]; then
    echo ""
    echo "========================================================================"
    echo "BUILDIFIER RESULTS"
    echo "========================================================================"
    echo ""

    # Extract buildifier errors from bazel output
    # Format 1: ./path/to/file.bzl:123: warning-type: Description
    # Format 2: path/to/file.bzl # reformat
    # Format 2 includes BUILD files with any suffix (e.g., BUILD, BUILD.bazel, BUILD.cosmos_tokenizer)
    BUILDIFIER_VIOLATIONS=$(grep -E '(^\./[^:]+:[0-9]+: |^[a-zA-Z0-9_/.-]+(BUILD[^ ]*|\.(bzl|bazel|sky)) # reformat)' "$BAZEL_OUTPUT_FILE" || true)

    if [[ -n "$BUILDIFIER_VIOLATIONS" ]]; then
        # Count both format types (line-number warnings and reformat violations)
        BUILDIFIER_VIOLATION_COUNT=$(echo "$BUILDIFIER_VIOLATIONS" | wc -l)
        # For files with violations, count unique files from both formats
        DETAILED_VIOLATIONS=$(echo "$BUILDIFIER_VIOLATIONS" | grep -E '^\./[^:]+:[0-9]+: ' || true)
        REFORMAT_VIOLATIONS=$(echo "$BUILDIFIER_VIOLATIONS" | grep -E '^[a-zA-Z0-9_/.-]+(BUILD[^ ]*|\.(bzl|bazel|sky)) # reformat' || true)

        FILES_COUNT=0
        if [[ -n "$DETAILED_VIOLATIONS" ]]; then
            FILES_COUNT=$(echo "$DETAILED_VIOLATIONS" | cut -d: -f1 | sort -u | wc -l)
        fi
        if [[ -n "$REFORMAT_VIOLATIONS" ]]; then
            REFORMAT_FILES=$(echo "$REFORMAT_VIOLATIONS" | sed 's/ # reformat$//' | wc -l)
            FILES_COUNT=$((FILES_COUNT + REFORMAT_FILES))
        fi
        FILES_WITH_BUILDIFIER_VIOLATIONS=$FILES_COUNT
    fi

    if [[ -z "$BUILDIFIER_VIOLATIONS" ]]; then
        echo "✓ No buildifier violations found!"
    else
        echo "Total: $FILES_WITH_BUILDIFIER_VIOLATIONS file(s) with $BUILDIFIER_VIOLATION_COUNT violation(s)"
        echo "$BUILDIFIER_VIOLATIONS"

        # Add to total violations count
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + BUILDIFIER_VIOLATION_COUNT))
    fi
fi

# Overall summary
echo ""
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""

TOTAL_FILES_WITH_VIOLATIONS=$((FILES_WITH_RUFF_VIOLATIONS + FILES_WITH_MYPY_VIOLATIONS + FILES_WITH_BUILDIFIER_VIOLATIONS))

if [[ $TOTAL_VIOLATIONS -eq 0 ]]; then
    printf '%b✓ All linting checks passed!%b\n' "${GREEN}" "${NC}"
else
    printf '%bTotal: %d file(s) with %d violation(s)%b\n' "${RED}" "$TOTAL_FILES_WITH_VIOLATIONS" "$TOTAL_VIOLATIONS" "${NC}"
    if [[ $RUFF_VIOLATIONS -gt 0 ]]; then
        printf '%b  - ruff: %d file(s), %d violation(s)%b\n' "${RED}" "$FILES_WITH_RUFF_VIOLATIONS" "$RUFF_VIOLATIONS" "${NC}"
    fi
    if [[ "$MYPY" == "true" && $MYPY_VIOLATIONS -gt 0 ]]; then
        printf '%b  - mypy: %d file(s), %d violation(s)%b\n' "${RED}" "$FILES_WITH_MYPY_VIOLATIONS" "$MYPY_VIOLATIONS" "${NC}"
    fi
    if [[ "$BUILDIFIER" == "true" && $BUILDIFIER_VIOLATION_COUNT -gt 0 ]]; then
        printf '%b  - buildifier: %d file(s), %d violation(s)%b\n' "${RED}" "$FILES_WITH_BUILDIFIER_VIOLATIONS" "$BUILDIFIER_VIOLATION_COUNT" "${NC}"
    fi
fi
echo ""

# Apply patches if requested
if [[ "$APPLY_PATCHES" == "true" ]]; then
    echo ""
    echo "========================================================================"
    echo "APPLYING AUTO-FIX PATCHES"
    echo "========================================================================"
    echo ""

    # Extract patch files from build events using jq
    if ! command -v jq &> /dev/null; then
        echo "Warning: jq not found, cannot extract patch files" >&2
        echo "Install jq to enable auto-fix: apt-get install jq" >&2
        exit 1
    fi

    # Parse build events to find .patch files
    ALL_PATCH_FILES=$(jq -r '
        select(.namedSetOfFiles.files != null) |
        .namedSetOfFiles.files[] |
        select(.name | endswith(".patch")) |
        if .pathPrefix then
            (.pathPrefix | join("/")) + "/" + .name
        else
            .name
        end
    ' "$BUILD_EVENTS_FILE" 2>/dev/null | sort -u || true)

    # Filter out empty patch files
    NON_EMPTY_PATCHES=()
    while IFS= read -r patch_file; do
        [[ -z "$patch_file" ]] && continue
        full_path="$WORKSPACE_ROOT/$patch_file"
        if [[ -f "$full_path" && -s "$full_path" ]]; then
            NON_EMPTY_PATCHES+=("$patch_file")
        fi
    done <<< "$ALL_PATCH_FILES"

    if [[ ${#NON_EMPTY_PATCHES[@]} -eq 0 ]]; then
        echo "No patches to apply."
    else
        PATCH_COUNT=${#NON_EMPTY_PATCHES[@]}

        echo "Found $PATCH_COUNT non-empty patch(es):"
        printf '  - %s\n' "${NON_EMPTY_PATCHES[@]}"
        echo ""

        # Apply the patches
        echo "Applying $PATCH_COUNT patch(es)..."
        ERRORS=0
        APPLIED=0
        for patch_file in "${NON_EMPTY_PATCHES[@]}"; do
            full_path="$WORKSPACE_ROOT/$patch_file"
            if patch -p1 -d "$WORKSPACE_ROOT" < "$full_path" 2>&1; then
                APPLIED=$((APPLIED + 1))
                echo "✓ Applied $(basename "$patch_file")"
            else
                echo "✗ Failed to apply $(basename "$patch_file")" >&2
                ERRORS=$((ERRORS + 1))
            fi
        done

        if [[ $ERRORS -gt 0 ]]; then
            echo "" >&2
            echo "✗ $ERRORS patch(es) failed to apply." >&2
            exit 1
        else
            echo "✓ Successfully applied all $APPLIED patch(es)!"
        fi
    fi

    if [[ "$BUILDIFIER" == "true" ]]; then
        set +e
        "${DAZEL_CMD[@]}" run //build/lint:buildifier
        BUILDIFIER_FIX_EXIT=$?
        set -e
        if [[ $BUILDIFIER_FIX_EXIT -ne 0 ]]; then
            echo "✗ Failed to run buildifier" >&2
            BAZEL_EXIT_CODE=$BUILDIFIER_FIX_EXIT
            exit 1
        else
            echo "✓ Successfully ran buildifier"
        fi
    fi
fi

# Exit with error only if FAIL_ON_VIOLATION is true and there were issues
if [[ "$FAIL_ON_VIOLATION" == "true" ]]; then
    # In fail-on-violation mode, exit with error if there were build failures or violations
    if [[ $BAZEL_EXIT_CODE -ne 0 || $TOTAL_VIOLATIONS -gt 0 ]]; then
        exit 1
    fi
fi

exit 0
