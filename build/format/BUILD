load("@aspect_rules_lint//format:defs.bzl", "format_multirun")

format_multirun(
    name = "format_multirun",
    jobs = 4,  # Number of parallel threads (corresponds to CPU cores) to use for formatting
    print_command = True,
    python = "@aspect_rules_lint//format:ruff",
    starlark = "@buildifier_prebuilt//:buildifier",
    starlark_check_args = [
        "-mode=check",
        "**/BUILD.*",
    ],
    starlark_fix_args = [
        "-mode=fix",
        "**/BUILD.*",
    ],
    visibility = ["//:__pkg__"],
)
