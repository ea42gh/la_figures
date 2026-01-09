# Interop

la_figures accepts Julia-style inputs via PythonCall/PyCall.

- Julia Symbols are normalized (e.g., `:tight`).
- Julia arrays are converted to nested Python lists when possible.
- Non-numeric entries may need explicit conversion before passing to Python.

## PythonCall sketch

```julia
using PythonCall
la = pyimport("la_figures")
A = [1 2; 3 4]
spec = la.ge_tbl_spec(A)
```

See `JULIA_INTEROP.md` in the repo root for concrete usage.
