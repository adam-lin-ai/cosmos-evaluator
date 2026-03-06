load("@aspect_rules_lint//format:defs.bzl", "format_multirun")
load("@pypi//:requirements.bzl", "requirement")
load("@rules_mypy//mypy:mypy.bzl", "mypy_cli")

format_multirun(
    name = "buildifier",
    jobs = 4,  # Number of parallel threads (corresponds to CPU cores) to use for formatting
    print_command = True,
    starlark = "@buildifier_prebuilt//:buildifier",
    starlark_check_args = [
        "-warnings=-constant-glob,-module-docstring",
        "-lint=warn",
        "-mode=check",
        "**/BUILD.*",
    ],
    starlark_fix_args = [
        "-lint=fix",
        "-mode=fix",
        "**/BUILD.*",
    ],
    visibility = ["//:__pkg__"],
)

mypy_cli(
    name = "mypy_cli",
    mypy_requirement = requirement("mypy"),
    python_version = "3.13",
    deps = [requirement("pydantic")],
)
