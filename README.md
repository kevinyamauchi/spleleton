# spleleton

[![License](https://img.shields.io/pypi/l/spleleton.svg?color=green)](https://github.com/kevinyamauchi/spleleton/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/spleleton.svg?color=green)](https://pypi.org/project/spleleton)
[![Python Version](https://img.shields.io/pypi/pyversions/spleleton.svg?color=green)](https://python.org)
[![CI](https://github.com/kevinyamauchi/spleleton/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinyamauchi/spleleton/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kevinyamauchi/spleleton/branch/main/graph/badge.svg)](https://codecov.io/gh/kevinyamauchi/spleleton)

spleleton

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork kevinyamauchi/spleleton --clone
# or just
# gh repo clone kevinyamauchi/spleleton
cd spleleton
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint and type-check files (using [prek](https://prek.j178.dev/)):

```sh
uv run prek run --all-files
```

To run the hooks automatically on every commit:

```sh
uv run prek install
```
