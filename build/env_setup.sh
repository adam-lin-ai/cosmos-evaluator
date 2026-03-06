#!/bin/bash

# This script is used to setup the build environment for Cosmos Evaluator.
# Supports both bash and zsh shells.

GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m' # No Color

# Minimum recommended versions
DOCKER_MIN_VERSION="28.1.1"
GIT_MIN_VERSION="2.49.0"
NV_DRIVER_MIN_VERSION="570.124.06"
NV_CONTAINER_TOOLKIT_MIN_VERSION="1.16.2"
PYTHON_MIN_VERSION="3.5.0"

# Completion script paths (set after COSMOS_EVALUATOR_ROOT is defined)
# Note: Don't use BAZEL_ prefix as it gets unset by the cleanup loop in _dazel_installation_check
COMPLETION_SCRIPT_BAZEL=""
COMPLETION_SCRIPT_DAZEL=""

# Detect which shell is being used
if [[ -n "$ZSH_VERSION" ]]; then
    _IS_ZSH=true
    _IS_BASH=false
elif [[ -n "$BASH_VERSION" ]]; then
    _IS_ZSH=false
    _IS_BASH=true
else
    _IS_ZSH=false
    _IS_BASH=false
fi

# Check if script is being sourced (not executed directly)
if $_IS_BASH; then
    _SCRIPT_SOURCE=${BASH_SOURCE[0]}

    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        echo "ERROR: This script should be sourced into the current shell.  Use the following syntax:"
        echo ""
        echo "    . build/env_setup.sh"
        echo ""
        exit 1
    fi
elif $_IS_ZSH; then
    _SCRIPT_SOURCE=${(%):-%N}

    # When sourced, ZSH_EVAL_CONTEXT ends with ":file" (e.g., "toplevel:file")
    # When executed directly, it's just "toplevel"
    if [[ "${ZSH_EVAL_CONTEXT:-}" != *:file ]]; then
        echo "ERROR: This script should be sourced into the current shell.  Use the following syntax:"
        echo ""
        echo "    . build/env_setup.sh"
        echo ""
        exit 1
    fi
else
    _SCRIPT_SOURCE=${0}
fi

# Export variables
# Git-based approach to find the Cosmos Evaluator root directory
COSMOS_EVALUATOR_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$(cd "$(dirname "${_SCRIPT_SOURCE}")/.." && pwd)")"
export COSMOS_EVALUATOR_ROOT

# Set completion script paths now that COSMOS_EVALUATOR_ROOT is defined
COMPLETION_SCRIPT_BAZEL="$COSMOS_EVALUATOR_ROOT/build/dazel/bazel_complete.sh"
COMPLETION_SCRIPT_DAZEL="$COSMOS_EVALUATOR_ROOT/build/dazel/dazel_complete.sh"

# Define functions
function dazel {
    $COSMOS_EVALUATOR_ROOT/build/dazel/dazel.py "$@"
}

if $_IS_BASH; then
    # Exporting functions is a bash only feature
    # For zsh, we need to ensure the function is available in subshells by adding it to the fpath or using autoload
    export -f dazel
fi

function _cosmos_evaluator_help() {
    cat <<EOF

Cosmos Evaluator build environment configured.

The following functions have been added to your environment:

  dazel                 Run bazel inside a Docker container

  Examples:

  dazel build //...     - Build all targets in Cosmos Evaluator
  dazel test //...      - Run all tests in Cosmos Evaluator

EOF
}

function _dazel_installation_check() {
    if ! command -v dazel &> /dev/null; then
        errors+=("Dazel installation unsuccessful.")
    else
        # Clear dazel commands cache to force refresh (works in both bash and zsh)
        unset _DAZEL_COMMANDS_CACHE 2>/dev/null || true

        if $_IS_BASH; then
            # Enable bash tab completion for dazel
            if [[ -f "$COMPLETION_SCRIPT_BAZEL" && -r "$COMPLETION_SCRIPT_BAZEL" ]]; then
                # Remove any existing completion registrations to allow clean reload
                complete -r dazel 2>/dev/null || true
                complete -r bazel 2>/dev/null || true
                complete -r ibazel 2>/dev/null || true

                # Unset all bazel completion functions to force reload
                # This ensures that any changes to the completion script are picked up
                for func in $(declare -F | grep -o '_bazel[^ ]*' 2>/dev/null); do
                    unset -f "$func" 2>/dev/null || true
                done

                # Unset bazel completion variables to force reload
                for var in $(compgen -v | grep '^BAZEL_' 2>/dev/null); do
                    unset "$var" 2>/dev/null || true
                done

                # Always re-source the completion script to pick up any updates
                if ! source "$COMPLETION_SCRIPT_BAZEL"; then
                    warnings+=("Failed to source bazel tab completion script.")
                fi

                # Source the custom dazel completion wrapper
                if [[ -f "$COMPLETION_SCRIPT_DAZEL" && -r "$COMPLETION_SCRIPT_DAZEL" ]]; then
                    if ! source "$COMPLETION_SCRIPT_DAZEL"; then
                        warnings+=("Failed to source dazel completion wrapper.")
                    fi
                fi

                # Register completion for dazel using our custom wrapper
                if declare -F _dazel_complete_wrapper >/dev/null 2>&1; then
                    complete -F _dazel_complete_wrapper -o nospace dazel
                elif declare -F _bazel__complete >/dev/null 2>&1; then
                    # Fallback to standard Bazel completion if wrapper not available
                    complete -F _bazel__complete -o nospace dazel
                else
                    warnings+=("Tab completion function not found.")
                fi
            else
                warnings+=("Tab completion script $COMPLETION_SCRIPT_BAZEL not found.")
            fi
        elif $_IS_ZSH; then
            # Enable zsh tab completion for dazel
            # Check if zsh completion system is initialized
            if ! typeset -f compdef >/dev/null 2>&1; then
                # Initialize zsh completion system if not already done
                autoload -Uz +X compinit 2>/dev/null && compinit -u 2>/dev/null
                autoload -Uz +X bashcompinit 2>/dev/null && bashcompinit -u 2>/dev/null
            fi

            # Source zsh-specific completion script if it exists
            if [[ -f "$COMPLETION_SCRIPT_BAZEL" && -r "$COMPLETION_SCRIPT_BAZEL" ]]; then
                # Remove any existing completion registrations to allow clean reload
                compdef -d dazel 2>/dev/null
                compdef -d bazel 2>/dev/null
                compdef -d ibazel 2>/dev/null

                # Unset all bazel completion functions to force reload
                for func in ${(k)functions[(I)_bazel*]}; do
                    unset -f "$func" 2>/dev/null
                done

                # Unset bazel completion variables to force reload
                for var in ${(k)parameters[(I)BAZEL_*]}; do
                    unset "$var" 2>/dev/null
                done

                if ! source "$COMPLETION_SCRIPT_BAZEL"; then
                    warnings+=("Failed to source bazel completion script for zsh.")
                fi
            fi
        fi

        echo -e "${GREEN}\u2714 Dazel successfully set up.${NC}"
    fi
}

function _compare_versions() {
    local min_version="$1"
    local current_version="$2"

    # Split versions into arrays (portable across bash/zsh)
    local -a min_version_parts current_version_parts
    if [[ -n "$ZSH_VERSION" ]]; then
        # zsh: use read -A and split on IFS
        IFS='.' read -r -A min_version_parts <<< "$min_version"
        IFS='.' read -r -A current_version_parts <<< "$current_version"
    else
        # bash: use read -a
        IFS='.' read -r -a min_version_parts <<< "$min_version"
        IFS='.' read -r -a current_version_parts <<< "$current_version"
    fi

    for i in 0 1 2; do
        local min=${min_version_parts[$i]:-0}
        local current=${current_version_parts[$i]:-0}
        if (( 10#$current > 10#$min )); then
            return 0  # current > min, version is OK
        elif (( 10#$current < 10#$min )); then
            return 1  # current < min, version is not OK
        fi
    done
    return 0  # versions are equal, OK
}

function _git_installation_check() {
    if ! command -v git &> /dev/null; then
        errors+=("Git is not installed. Please install git before continuing.")
    else
        git_version=$(git --version | awk '{print $3}')

        if _compare_versions "$GIT_MIN_VERSION" "$git_version"; then
            version_ok=true
        else
            version_ok=false
        fi

        if ! $version_ok; then
            warnings+=("Git version $git_version is less than the recommended version $GIT_MIN_VERSION")
        else
            echo -e "${GREEN}\u2714 Git version $git_version meets recommended requirement.${NC}"
        fi
    fi
}

function _docker_installation_check() {
    if ! command -v docker &> /dev/null; then
        errors+=("Docker is not installed. Please install Docker before continuing.")
    else
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//g')

        if _compare_versions "$DOCKER_MIN_VERSION" "$docker_version"; then
            version_ok=true
        else
            version_ok=false
        fi
        if ! $version_ok; then
            warnings+=("Docker version $docker_version is less than the recommended version $DOCKER_MIN_VERSION")
        else
            echo -e "${GREEN}\u2714 Docker version $docker_version meets recommended requirement.${NC}"
        fi
    fi
}

function _nvidia_driver_installation_check() {
    if ! command -v nvidia-smi &> /dev/null; then
        errors+=("Nvidia GPU driver is not installed. Please install it before continuing.")
    else
         nv_driver_version=$(modinfo --field=version nvidia)

         if _compare_versions "$NV_DRIVER_MIN_VERSION" "$nv_driver_version"; then
             version_ok=true
         else
             version_ok=false
         fi

         if ! $version_ok; then
             warnings+=("Nvidia GPU driver version $nv_driver_version is less than the recommended version $NV_DRIVER_MIN_VERSION, running GPU-enabled tests/binaries locally may fail")
         else
             echo -e "${GREEN}\u2714 Nvidia GPU driver version $nv_driver_version meets recommended requirement.${NC}"
         fi
    fi
}

function _nvidia_container_toolkit_installation_check() {
    if ! command -v nvidia-container-toolkit &> /dev/null; then
        errors+=("Nvidia container toolkit is not installed. Please install it before continuing.")
    else
        # Get the version of nvidia-container-toolkit assuming it is the last part of the output line
        toolkit_version=$(nvidia-container-toolkit --version 2>/dev/null | head -n1 | awk '{print $NF}')
        if [[ -z "$toolkit_version" ]]; then
            warnings+=("Could not determine nvidia-container-toolkit version.")
        else
            if _compare_versions "$NV_CONTAINER_TOOLKIT_MIN_VERSION" "$toolkit_version"; then
                version_ok=true
            else
                version_ok=false
            fi

            if ! $version_ok; then
                warnings+=("Nvidia container toolkit version $toolkit_version is less than the recommended version $NV_CONTAINER_TOOLKIT_MIN_VERSION, running GPU-enabled tests/binaries locally may fail")
            else
                echo -e "${GREEN}\u2714 Nvidia container toolkit version $toolkit_version meets recommended requirement.${NC}"
            fi
        fi
    fi
}

function _python_installation_check() {
    # Dazel uses #!/usr/bin/env python3, so we MUST have a python3 executable
    if ! command -v python3 &> /dev/null; then
        if command -v python &> /dev/null; then
            # Check if 'python' is actually Python 3
            local python_major
            python_major=$(python -c 'import sys; print(sys.version_info[0])' 2>/dev/null)
            if [[ "$python_major" == "3" ]]; then
                errors+=("'python3' executable not found, but 'python' points to Python 3. Dazel requires a 'python3' executable (its shebang is #!/usr/bin/env python3). Please create a symlink: sudo ln -s \$(which python) /usr/local/bin/python3")
            else
                errors+=("'python3' executable not found. Only Python 2.x is available. Dazel requires Python $PYTHON_MIN_VERSION or later with a 'python3' executable.")
            fi
        else
            errors+=("Python 3 is not installed. Dazel requires Python $PYTHON_MIN_VERSION or later with a 'python3' executable.")
        fi
        return
    fi

    # Get Python version (e.g., "3.10.12")
    local python_version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null)

    if [[ -z "$python_version" ]]; then
        errors+=("Could not determine Python version. Dazel requires Python $PYTHON_MIN_VERSION or later.")
    elif _compare_versions "$PYTHON_MIN_VERSION" "$python_version"; then
        echo -e "${GREEN}\u2714 Python version $python_version meets minimum requirement.${NC}"
    else
        errors+=("Python version $python_version is less than the minimum required version $PYTHON_MIN_VERSION. Dazel will not work.")
    fi
}

function _cosmos_evaluator_install_build_requirements() {
    local -a warnings=()
    local -a errors=()

    _python_installation_check
    _docker_installation_check
    _git_installation_check

    # Only run NVIDIA checks if a GPU exists on the machine
    if command -v lspci &> /dev/null && lspci 2>/dev/null | grep -iq nvidia; then
        _nvidia_driver_installation_check
        _nvidia_container_toolkit_installation_check
    else
        if command -v lspci &> /dev/null; then
            echo -e "${YELLOW}WARNING: No NVIDIA GPU detected. Skipping GPU-related checks.${NC}"
        else
            echo -e "${YELLOW}WARNING: lspci not available. Skipping GPU detection and related checks.${NC}"
        fi
    fi

    # Only set up dazel if there are no critical errors (like missing Python)
    if [[ ${#errors[@]} -eq 0 ]]; then
        _dazel_installation_check
    else
        echo -e "${RED}\u2718 Dazel was NOT installed due to missing requirements.${NC}"
    fi

    # Check for errors
    for e in "${errors[@]}"; do
        echo -e "${RED}ERROR: $e${NC}"
    done

    # Errors occurred, return non-zero exit code
    if [[ ${#errors[@]} -ne 0 ]]; then
        return 1
    fi

    _cosmos_evaluator_help

    for w in "${warnings[@]}"; do
        echo -e "${YELLOW}WARNING: $w${NC}"
    done

    return 0
}

# Script's main body that is executed when script is sourced
_cosmos_evaluator_install_build_requirements || echo -e "${RED}Build environment setup failed.${NC}"

# Remove temporary variables from user environment
unset GREEN YELLOW RED NC DOCKER_MIN_VERSION GIT_MIN_VERSION NV_DRIVER_MIN_VERSION NV_CONTAINER_TOOLKIT_MIN_VERSION PYTHON_MIN_VERSION COMPLETION_SCRIPT_BAZEL COMPLETION_SCRIPT_DAZEL _IS_BASH _IS_ZSH _SCRIPT_SOURCE
