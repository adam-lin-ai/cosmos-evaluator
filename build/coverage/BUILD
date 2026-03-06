load("@pypi//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "lcov_to_cobertura",
    srcs = ["lcov_to_cobertura.py"],
    tags = ["no-mypy"],
    deps = [
        requirement("lcov_cobertura"),
    ],
)
