#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
.venv/bin/python -m streamlit run app.py
