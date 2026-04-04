# Publishing to PyPI

This guide explains how to build and publish `sicifus` to the Python Package Index (PyPI).

## Prerequisites

You need to have `build` and `twine` installed:

```bash
pip install build twine
```

## 1. Update Version

Before publishing, make sure to update the version number in `pyproject.toml`:

```toml
[project]
name = "sicifus"
version = "0.3.0"  # <--- Update this
```

## 2. Build the Package

Run the following command from the root of the repository to generate the distribution files (source archive and wheel):

```bash
python -m build
```

This will create a `dist/` directory containing:
-   `sicifus-x.y.z.tar.gz` (Source distribution)
-   `sicifus-x.y.z-py3-none-any.whl` (Wheel)

## 3. Check the Package

It's good practice to run checks on your distribution files to ensure they are valid:

```bash
twine check dist/*
```

## 4. Upload to TestPyPI (Optional but Recommended)

TestPyPI is a separate instance of PyPI for testing. It allows you to verify that everything looks correct before the real release.

1.  Register an account at [test.pypi.org](https://test.pypi.org/).
2.  Create an API token.
3.  Upload:

```bash
twine upload --repository testpypi dist/*
```

4.  Try installing it:

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps sicifus
```

## 5. Upload to PyPI (Production)

1.  Register an account at [pypi.org](https://pypi.org/).
2.  Create an API token.
3.  Upload:

```bash
twine upload dist/*
```

You will be prompted for your username (`__token__`) and your API token as the password.

## 6. Verify Installation

Wait a few minutes, then try installing your package from the real PyPI:

```bash
pip install sicifus
```
