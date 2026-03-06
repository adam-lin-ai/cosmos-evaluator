#!/usr/bin/env bash
#
# Custom completion wrapper for dazel that adds support for the 'lint' and 'fix' commands
#

# This function wraps _bazel__complete to add custom dazel commands
_dazel_complete_wrapper() {
    local cur cword
    _init_completion -n : || return

    # If we're completing the first argument (the command)
    if [[ $cword -eq 1 ]]; then
        # Get dazel's commands dynamically and add our custom ones
        local -a dazel_commands
        # Cache dazel commands to avoid repeated calls
        if [[ -z "${_DAZEL_COMMANDS_CACHE:-}" ]] && command -v dazel &>/dev/null; then
            mapfile -t _DAZEL_COMMANDS_CACHE < <(dazel help 2>/dev/null | awk '/^  [a-z]/ && $1 != "bazel" {print $1}')
        fi
        dazel_commands=("${_DAZEL_COMMANDS_CACHE[@]}")

        # Add our custom commands lint and fix
        dazel_commands+=(lint fix)

        mapfile -t COMPREPLY < <(compgen -W "${dazel_commands[*]}" -- "$cur")

        return 0
    fi

    # If the command is 'lint' or 'fix', pretend it's 'build' for target completion
    if [[ ${COMP_WORDS[1]} == "lint" || ${COMP_WORDS[1]} == "fix" ]]; then
        # Just call bazel completion with 'build' instead
        # Save original and modify
        local saved_words1="${COMP_WORDS[1]}"
        COMP_WORDS[1]="build"

        # Call bazel completion
        _bazel__complete

        # Restore
        COMP_WORDS[1]="$saved_words1"
        return 0
    fi

    # For all other commands, use standard Bazel completion
    _bazel__complete
}
