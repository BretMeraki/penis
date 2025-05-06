# AI Strategic Co-Founder Directive

**Role:**
You are to operate as a strategic co-founder and operator for this project, with the mindset and standards of someone scaling a $10M+ valuation technology company. Your actions, recommendations, and code changes should reflect the following principles:

- **Pragmatism First:** Prioritize solutions that are practical, robust, and deliver real value to users and the business. Avoid over-engineering; focus on what moves the needle.
- **Founder/Operator Mindset:** Make decisions as if you are responsible for the success, growth, and sustainability of the company. Consider technical debt, user experience, and business impact in every action.
- **Scale-Ready:** Ensure all code, architecture, and processes are designed to scale efficiently—technically, operationally, and organizationally.
- **Bias for Action:** When in doubt, take decisive action. Ship improvements, fix blockers, and iterate quickly. Document rationale for major decisions.
- **User and Business Alignment:** Every change should either improve user experience, increase reliability, or drive business value. Ruthlessly prioritize.
- **Transparency and Documentation:** Clearly document all major changes, tradeoffs, and strategic decisions in this file and in code comments where relevant.
- **Continuous Improvement:** Always look for ways to optimize, automate, and future-proof the product and team workflows.

---

*This directive sets the tone for all AI-driven actions in this project. The AI is empowered to act as a true co-founder/operator, not just a coding assistant.*

---

# Forest OS Handover Document

## Project Overview

Forest OS is a modular, extensible personal growth assistant built with a FastAPI backend and a Streamlit frontend. The system is designed for robust, testable, and user-friendly operation, with a focus on personal development workflows, skill trees (HTA), and snapshot-based state management.

### Key Components

- **Backend:** FastAPI app (`forest_app/core/main.py`) with modular routers, dependency injection (via `dependency_injector`), and feature flag support.
- **Frontend:** Streamlit app (`forest_app/front_end/streamlit_app.py`) with modular UI, authentication, onboarding, and HTA visualization.
- **Modules:** Cognitive, resource, relational, memory, harmonic, and snapshot_flow modules, each encapsulating a domain-specific engine or manager.
- **Persistence:** SQLite database with migration support, managed via the `forest_app/persistence` and `forest_app/snapshot` modules.
- **Integrations:** LLM (Large Language Model) and MCP (external service) integrations, with robust fallback/dummy implementations for local development.
- **Testing:** Pytest-based test suite for backend, frontend, and HTA lifecycle.

---

## Work Done To Date

- **Full modularization** of backend and frontend, with clear separation of concerns.
- **Dependency injection** via `forest_app/core/containers.py` for all major services and engines.
- **Feature flag system** for toggling experimental or optional features.
- **Robust error handling** and logging throughout backend and frontend.
- **Streamlit frontend** optimized for user experience, with onboarding, authentication, and HTA visualization.
- **API client abstraction** for frontend-backend communication, with consistent error and data handling.
- **Snapshot and HTA (Hierarchical Task Analysis) engines** for tracking user progress and goals.
- **Comprehensive test suite** for backend endpoints, frontend import/caching, and HTA lifecycle.
- **Sentry integration** for error monitoring (configurable via environment variables).
- **Configuration management** via `forest_app/config/settings.py` and `.env` file.
- **Database initialization and migration** logic included in startup routines.
- **Fallback/dummy implementations** for all major services to ensure the app runs even if some modules fail to import.

---

## Development Quirks & Key Notes

### General

- **Strict modularity:** All new features should be implemented as modules or services and registered in the DI container.
- **Dummy fallbacks:** If a module fails to import, a dummy implementation is used. Always check logs for warnings about dummy services.
- **Feature flags:** Use feature flags for any experimental or optional features. Check `forest_app/core/feature_flags.py`.
- **Configuration:** All config should be centralized in `settings.py` and loaded via the DI container.
- **Logging:** Logging is set up at the top of most files. Use the provided logger for all debug/info/error output.

### Backend

- **Routers:** All API endpoints are registered via routers in `forest_app/routers/` and included in `main.py`.
- **Dependency injection:** All services, engines, and processors are provided via the DI container. Never instantiate these directly.
- **Database:** Uses SQLite by default. Initialization is handled at startup. Migrations are supported.
- **Security:** Security dependencies are initialized at startup. User model must have an `email` field.

### Frontend

- **Streamlit:** Only one call to `st.set_page_config` is allowed (enforced by tests).
- **No deprecated caching:** Do not use `@st.cache`; use `@st.cache_data` or `@st.cache_resource` if needed.
- **Session state:** All user/session data is managed via `st.session_state` with consistent key usage.
- **API client:** All backend calls go through `api_client.py` for consistent error handling and logging.
- **Authentication:** Handled in `auth_ui.py`, with robust error handling and session state management.
- **Onboarding:** Modularized in `onboarding_ui.py`.

### Testing

- **Tests are in `tests/`** and cover backend endpoints, frontend import/caching, and HTA lifecycle.
- **Frontend tests** check for import errors, duplicate `set_page_config` calls, and deprecated caching.
- **Backend tests** use FastAPI's `TestClient` and dependency overrides for user authentication.
- **HTA lifecycle tests** use dummy models and async test cases to simulate user progress and tree evolution.

### Robustness & User Experience

- **Error handling:** All user-facing errors are displayed in the UI or returned in API responses.
- **Fallbacks:** If a feature or service fails, the app will use a dummy and log a warning, but will not crash.
- **User experience:** The frontend is designed for clarity, with onboarding, clear error messages, and visualizations.
- **Tuning:** All modules and services are designed to be easily tunable via config or feature flags.

---

## Recommendations for New Developers/AI

- **Always check logs** for warnings about dummy services or failed imports.
- **Use the DI container** for all service/engine access.
- **Add new features as modules** and register them in the container.
- **Write tests** for all new endpoints and frontend features.
- **Keep user experience in mind**—optimize for clarity, error recovery, and visual feedback.
- **Document any new quirks** or workarounds in this handover file.

---

## Outstanding Issues / TODOs

- **Centralize constants:** Some constants are duplicated between frontend modules; consider centralizing.
- **Expand test coverage:** Some modules may lack direct tests; expand as needed.
- **Improve onboarding modularity:** Onboarding logic is split between files; consider further modularization.
- **Review feature flag usage:** Ensure all optional features are properly gated.
- **Refactor dummy fallbacks:** Consider more granular logging or user feedback when dummies are used.

---

This document should be updated as new features are added or quirks are discovered.

--- 