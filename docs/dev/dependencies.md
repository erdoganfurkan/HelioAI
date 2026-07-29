# Dependency policy

HelioAI follows [PHEP 3](https://doi.org/10.5281/zenodo.17794207), the PyHC adoption of
[SPEC 0](https://scientific-python.org/specs/spec-0000/).

## The rules

- **Python minor versions** are supported for **36 months** after release.
- **Upstream core Scientific Python packages** — `numpy`, `scipy`, `matplotlib`, `pandas`,
  `scikit-image`, `networkx`, `scikit-learn`, `xarray`, `ipython`, `zarr` — are supported
  for **24 months** after release.
- New versions of both are adopted within **6 months** of release.

In practice that means about three Python minor versions at any time. Today:
**3.12, 3.13 and 3.14**.

Compliance is checked in CI by the official
[PyHC PHEP 3 action](https://github.com/heliophysicsPy/pyhc-actions), so the policy is
enforced rather than merely written down.

## What this means when you add a dependency

Use `>=` floors. Upper bounds and exact pins are the exception, not the default:

```toml
"numpy>=2.1",        # good
"numpy<2",           # needs a very good reason and a tracking issue
"scipy==1.15.0",     # almost never
```

PHEP 3 permits an upper bound "only when absolutely necessary", and requires an issue to be
opened at the same time to remove it.

!!! warning "Note that the floor matters too"
    A floor older than 24 months is itself a warning: it means you claim to support a
    version the ecosystem has moved past. Bumping the floor is part of routine maintenance,
    not a breaking change.

### Current exceptions

| Pin | Why | Removal |
|---|---|---|
| `mcp>=1.0,<2` | mcp 2.0 is a breaking rewrite of both the client and server APIs HelioAI uses | tracked; lifts with the `mcp_server.py` migration |

There is also a uv-only constraint, `torch>=2.6`, in `[tool.uv]`. It exists purely so the
lockfile resolves a torch with cp313/cp314 wheels — left alone the universal resolve
settles on 2.4.1, which has none. It is deliberately **not** a `dependencies` entry:
HelioAI never imports torch, and PyPI users should resolve it themselves.

## Lifting a cap is not a local change

Removing an upper bound re-resolves the **entire** graph, not just the package you touched.
Dropping `transformers<5.0` here also let `mcp` jump from 1.x to 2.0, which broke the web
server at startup.

So, when you lift a cap:

1. Read the **whole** `uv lock` diff, not just the dependency you meant to change.
2. Run the application, not only the tests — `helioai serve --web` caught that one before
   CI did.
3. Expect the lint to move too. Pinning ruff's rule set explicitly (rather than inheriting
   its shifting defaults) exists for the same reason.

## Keeping the environment sane

`uv sync --extra X` **purges** extras you did not list. Always combine:

```bash
uv sync --extra dev --extra solarmach
```

Otherwise pytest quietly disappears from your environment, and the failure you get next is
confusing rather than obvious.
