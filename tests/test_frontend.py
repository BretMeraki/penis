import importlib
import sys
import os
import pytest

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_streamlit_app_import():
    try:
        import forest_app.front_end.streamlit_app
    except Exception as e:
        pytest.fail(f"streamlit_app.py import failed: {e}")


def test_no_duplicate_set_page_config():
    with open("forest_app/front_end/streamlit_app.py") as f:
        lines = f.readlines()
    count = sum(1 for line in lines if "st.set_page_config" in line)
    assert count == 1, f"Expected 1 st.set_page_config call, found {count}"


def test_no_st_cache():
    with open("forest_app/front_end/streamlit_app.py") as f:
        content = f.read()
    assert "@st.cache" not in content, "Deprecated @st.cache found in streamlit_app.py. Use @st.cache_data or @st.cache_resource." 