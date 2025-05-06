## Running the Test Environment

1. **Install test dependencies:**
   ```bash
   pip install -r requirements-test.txt
   ```

2. **Run all tests:**
   ```bash
   pytest
   ```

- Backend tests are in `tests/test_backend.py`.
- Frontend (Streamlit) import/caching tests are in `tests/test_frontend.py`.

If any test fails, check the error message for the exact bug to fix. 