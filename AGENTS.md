# OpenMC Development Guide

## Cursor Cloud specific instructions

### Overview
OpenMC is a Monte Carlo particle transport simulation code. It has a C++ core built with CMake and a Python API installed via pip.

### Environment variables (already configured in `~/.bashrc`)
- `OPENMC_CROSS_SECTIONS=$HOME/nndc_hdf5/cross_sections.xml` — path to NNDC HDF5 nuclear data
- `OPENMC_ENDF_DATA=$HOME/endf-b-vii.1` — path to ENDF/B-VII.1 distribution
- `PATH` includes `/workspace/build/bin` (for the `openmc` executable) and `$HOME/.local/bin` (for pytest, etc.)

### Building (after update script runs)
The C++ library must be rebuilt if C++ source or CMake files change:
```
cd /workspace/build && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DOPENMC_USE_OPENMP=on -DHDF5_PREFER_PARALLEL=OFF .. && make -j$(nproc)
```
The build step copies `libopenmc.so` into `openmc/lib/` automatically, so the Python ctypes bindings pick it up without reinstalling the Python package.

### Running tests
- **C++ unit tests**: `cd /workspace/build && ctest` (5 tests)
- **Python unit tests**: `pytest tests/unit_tests/` (~760 tests, ~70s)
- **Regression tests**: `pytest tests/regression_tests/` (many tests, can be slow)
- **Single regression test**: `pytest tests/regression_tests/<test_name>/`
- Known pre-existing failure: `test_export_to_hdf5[Pu]` in `tests/unit_tests/test_data_photon.py`

### Lint / format checking
- C++ format checking uses `clang-format` version 15 with the config in `.clang-format`
- Python style: see the [style guide](https://docs.openmc.org/en/latest/devguide/styleguide.html)
- No unified `lint` command; CI uses `cpp-linter/cpp-linter-action@v2`

### Gotchas
- The default C++ compiler (Clang 18) fails to link because it selects GCC 14 runtime which lacks `libstdc++.so`. Always pass `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++` to CMake.
- Nuclear cross-section data (~1 GB) is downloaded to `$HOME/nndc_hdf5/` and `$HOME/endf-b-vii.1/` and cached between sessions. If missing, run `bash tools/ci/download-xs.sh`.
- Skip optional features (DAGMC, libMesh, NCrystal, MPI, MCPL) unless specifically needed; tests for those are auto-skipped.
