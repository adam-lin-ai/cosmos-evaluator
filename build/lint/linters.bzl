load("@aspect_rules_lint//lint:ruff.bzl", "lint_ruff_aspect")
load("@pip_types//:types.bzl", "types")
load("@rules_mypy//mypy:mypy.bzl", "mypy")

mypy_aspect = mypy(
    mypy_cli = "@@//build/lint:mypy_cli",
    mypy_ini = "@@//:pyproject.toml",
    types = types,
)

ruff_aspect = lint_ruff_aspect(
    binary = Label("@aspect_rules_lint//lint:ruff_bin"),
    configs = [
        Label("//:pyproject.toml"),
    ],
)
