# Linting

Tool that checks code for linting violations and can automatically fixes some issues:
* Displays lint violation content to see what lines of code are causing the violations
* Some violations cannot be auto-fixed with the `--apply-patches` argument. Please manually fix any remaining violations.

**Recommendation: Run `dazel fix` before submitting an MR. Manually fix any violations that remain.**

## Usage

The linter can be run directly or with the convenience command `dazel lint`:

```text
Usage: linter.sh [OPTIONS] [TARGETS...]

Run linters on Bazel targets

Options:
  --apply-patches           Auto-fix violations by applying patches (default: false)
  --fail-on-violation       Fail the build when a linter warning is present (default: false)
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
  linter.sh --no-mypy                    # Run only ruff (skip mypy)
```

### Examples

```bash
# View lint violations (default behavior)
build/lint/linter.sh                 # Check all targets
build/lint/linter.sh //checks/...    # Check specific targets

# Fail with non-zero exit code if violations found (useful for CI/CD)
build/lint/linter.sh --fail-on-violation

# Auto-fix violations by applying patches
build/lint/linter.sh --apply-patches
build/lint/linter.sh --apply-patches //checks/...
```

### Convenience Commands

```bash
# Using dazel lint (shows violations without applying fixes)
dazel lint                    # Check all targets (//...)
dazel lint //checks/...       # Check targets under //checks/...

# Using dazel fix (applies patches and runs formatter)
dazel fix                     # Apply patches and format all targets
dazel fix //checks/...        # Apply patches and format only targets under //checks/...
```
