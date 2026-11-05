from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

@pytest.mark.parametrize("page",["Overview","Data & readiness","Train & freeze","Forecasts","Explanations","Performance"])
def test_empty_pages(page,tmp_path,monkeypatch):
    monkeypatch.setenv("SOCCER_DATA_DIR",str(tmp_path))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.error
