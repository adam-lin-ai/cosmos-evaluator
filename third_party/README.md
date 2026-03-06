# Third-Party Dependencies

This directory contains external third-party repositories and libraries used by the Cosmos Evaluator project.

| Name | Source | Version | Description |
|------|--------|---------|-------------|
| **cosmos_drive_dreams_toolkits** | [nv-tlabs/Cosmos-Drive-Dreams](https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-drive-dreams-toolkits) | `908ba8f6` | Toolkits for Cosmos-Drive-Dreams, Cosmos-Transfer1, and Cosmos-Transfer2, working with RDS-HQ format data |

## Conventions

1. Create an [`http_archive`](https://bazel.build/rules/lib/repo/http#http_archive) entry in the [MODULE.bazel](../MODULE.bazel) file
    * Set the [`strip_prefix`](https://bazel.build/rules/lib/repo/http#http_archive-strip_prefix) attribute to the deepest directory in the repository that contains all the code required (this ensures we only pull the minimum subset of the repo necessary)
    * Set the [`add_prefix`](https://bazel.build/rules/lib/repo/http#http_archive-add_prefix) attribute to `"//third_party/<repo_name>/path/to/deepest/directory"`
      * **NOTE**:  If the `repo_name` is redundant with the name of a root-level subdirectory (e.g. `detectron2/detectron2`), the `repo_name` should be dropped from the `add_prefix` prefix path
2. Create a `BUILD.<repo_name>` file in this `third_party/` directory and add BUILD target(s) for the code needed for use from the third party repo
3. Appropriately set the [`imports`](https://bazel.build/reference/be/python#py_library.imports) attribute to ensure that code internal to the third party repo is able to properly `import` its modules
