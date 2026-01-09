# la-figures

Linear algebra algorithms and descriptor objects for the itikz migration.

This package contains algorithmic traces/specs (e.g., eigen tables, GE traces)
but does not render or invoke TeX toolchains.

## Decorators

The convenience wrappers pass decorator specs through to `matrixlayout`. Use
selector helpers from `matrixlayout.formatting` (or re-exported via
`la_figures`) to target entries or vector rows.

## Julia interop

See `../JULIA_INTEROP.md` for PyCall/PythonCall usage patterns and a small
Julia façade module that re-exports the Python convenience wrappers under the
same top-level function names.

## Documentation

MkDocs configuration lives in `la_figures/mkdocs.yml` with content under
`la_figures/docs/`.

Build the docs:

```bash
cd la_figures
mkdocs build
```
