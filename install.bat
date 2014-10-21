:: install anaconda

@echo OFF

python --version 2>NUL
if not errorlevel 1 goto installNengo

reg Query "HKLM\Hardware\Description\System\CentralProcessor\0" | find /i "x32" > NUL && set OS=32BIT || set OS=64BIT

if %OS%==32BIT .\packages\windows\32_bit\Anaconda-2.1.0-Windows-x86.exe
if %OS%==64BIT .\packages\windows\64_bit\Anaconda-2.1.0-Windows-x86_64.exe

:: add the path to our environment without having to restart the bath file
:: This might be impossible
@powershell -NoProfile -ExecutionPolicy unrestricted -Command "$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ";" + [System.Environment]::GetEnvironmentVariable('Path','User')"

:: Given that it's impossible, maybe just have two scripts? One that checks if Python is installed and one that doesnt?

:: Wait, but what about installing pip and all that?
: installNengo
python --version 2>NUL
if errorlevel 1 goto errorNoPython

python setup.py develop

:errorNoPython
echo.
echo Error^: Python not installed