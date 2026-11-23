from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_app_and_constraint():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == '56,000'
    app.number_input[0].set_value(1.5).run(timeout=60)
    assert not app.exception
    assert any(c.value == 'MONITOR' for c in app.code)
