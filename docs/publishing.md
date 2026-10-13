# Publishing

Build and validate a release:

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

For the first release, log into PyPI as `wwzzz1998` and add a pending
[trusted publisher](https://pypi.org/manage/account/publishing/):

| Field | Value |
| --- | --- |
| PyPI project | `vlastudio` |
| Repository owner | `WwZzz` |
| Repository | `VLAStudio` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Merge the release changes into main, then publish the GitHub release `v0.2.0`.
The workflow validates and uploads the wheel and source distribution without
a stored API token. For later releases, update the version in `pyproject.toml`,
`src/vlastudio/__init__.py`, and `uv.lock`, then use the matching release tag.

Alternatively, configure a PyPI API token in your local credential store and run:

```bash
python -m twine upload dist/*
```

Never commit credentials. PyPI release files cannot be overwritten.
