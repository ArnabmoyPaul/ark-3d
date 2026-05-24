@echo off
REM Run this from D:\ark+3d (or wherever your project is)
REM Double-click it OR run from terminal

set ROOT=%~dp0
echo Project root: %ROOT%

REM Create folder structure
mkdir "%ROOT%ark_plus_3d" 2>nul
mkdir "%ROOT%ark_plus_3d\models" 2>nul
mkdir "%ROOT%ark_plus_3d\data" 2>nul
mkdir "%ROOT%ark_plus_3d\training" 2>nul
mkdir "%ROOT%ark_plus_3d\utils" 2>nul
mkdir "%ROOT%ark_plus_3d\configs" 2>nul
mkdir "%ROOT%ark_plus_3d\notebooks" 2>nul
mkdir "%ROOT%checkpoints" 2>nul
mkdir "%ROOT%data" 2>nul

REM Create __init__.py files
type nul > "%ROOT%ark_plus_3d\__init__.py"
type nul > "%ROOT%ark_plus_3d\models\__init__.py"
type nul > "%ROOT%ark_plus_3d\data\__init__.py"
type nul > "%ROOT%ark_plus_3d\training\__init__.py"
type nul > "%ROOT%ark_plus_3d\utils\__init__.py"
type nul > "%ROOT%ark_plus_3d\configs\__init__.py"

REM Move files to correct locations (only if not already there)
if exist "%ROOT%ark_plus.py"      move "%ROOT%ark_plus.py"      "%ROOT%ark_plus_3d\models\ark_plus.py"
if exist "%ROOT%swin3d.py"        move "%ROOT%swin3d.py"        "%ROOT%ark_plus_3d\models\swin3d.py"
if exist "%ROOT%dataset.py"       move "%ROOT%dataset.py"       "%ROOT%ark_plus_3d\data\dataset.py"
if exist "%ROOT%engine.py"        move "%ROOT%engine.py"        "%ROOT%ark_plus_3d\training\engine.py"
if exist "%ROOT%metrics.py"       move "%ROOT%metrics.py"       "%ROOT%ark_plus_3d\utils\metrics.py"
if exist "%ROOT%helpers.py"       move "%ROOT%helpers.py"       "%ROOT%ark_plus_3d\utils\helpers.py"
if exist "%ROOT%medmnist_3d.yaml" move "%ROOT%medmnist_3d.yaml" "%ROOT%ark_plus_3d\configs\medmnist_3d.yaml"
if exist "%ROOT%ark_3d_run.ipynb" move "%ROOT%ark_3d_run.ipynb" "%ROOT%ark_plus_3d\notebooks\ark_3d_run.ipynb"

echo.
echo Done! Final structure:
tree "%ROOT%ark_plus_3d" /F

pause
