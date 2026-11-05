@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create a virtual environment first: py -3.11 -m venv .venv
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
