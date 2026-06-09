# Build the edm_nonlinear_solver C++ extension for Python on Windows (MSVC + VS2022)
# Run this script from the ChroMo_PE directory.

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BUILD_DIR  = Join-Path $SCRIPT_DIR "build"

# Resolve Python: prefer venv adjacent to ChroMo_PE (../venv), else active python
$VENV_PYTHON = Join-Path $SCRIPT_DIR "..\venv\Scripts\python.exe"
if (Test-Path $VENV_PYTHON) {
    $PYTHON_EXE = (Resolve-Path $VENV_PYTHON).Path
} else {
    $PYTHON_EXE = (Get-Command python).Source
}
Write-Host "Using Python: $PYTHON_EXE"
$PYBIND11_DIR = & $PYTHON_EXE -c "import pybind11; print(pybind11.get_cmake_dir())"

Write-Host "== Configuring with CMake (Visual Studio 17 2022, x64) =="
New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null
cmake -S $SCRIPT_DIR -B $BUILD_DIR `
      -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_BUILD_TYPE=Release `
      "-Dpybind11_DIR=$PYBIND11_DIR" `
      "-DPython_EXECUTABLE=$PYTHON_EXE"

if (-not $?) { Write-Error "CMake configure failed"; exit 1 }

Write-Host "`n== Building =="
cmake --build $BUILD_DIR --config Release

if (-not $?) { Write-Error "CMake build failed"; exit 1 }

Write-Host "`n== Copying .pyd to ChroMo_PE root =="
$pyd = Get-ChildItem "$BUILD_DIR\Release" -Filter "edm_nonlinear_solver*.pyd" -ErrorAction SilentlyContinue |
       Select-Object -First 1
if (-not $pyd) {
    $pyd = Get-ChildItem $BUILD_DIR -Recurse -Filter "edm_nonlinear_solver*.pyd" -ErrorAction SilentlyContinue |
           Select-Object -First 1
}
if (-not $pyd) { Write-Error "Built .pyd not found - check build output above"; exit 1 }

Copy-Item $pyd.FullName -Destination (Join-Path $SCRIPT_DIR "edm_nonlinear_solver.pyd") -Force
Write-Host "Copied: $($pyd.Name) -> ChroMo_PE\edm_nonlinear_solver.pyd"
Write-Host "`n== Done! C++ solver library is ready. =="
