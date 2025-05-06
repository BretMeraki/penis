# front_end/streamlit_app.py (Refactored: Uses api_client, auth_ui, onboarding_ui)

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from forest_app.core.logging_tracking import log_once_per_session
from logging.handlers import RotatingFileHandler

# Set up rotating error log before any other imports
logfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'error.log'))
rotating_handler = RotatingFileHandler(logfile, maxBytes=1_000_000, backupCount=3)
rotating_handler.setLevel(logging.WARNING)
rotating_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
root_logger = logging.getLogger()
if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == logfile for h in root_logger.handlers):
    root_logger.addHandler(rotating_handler)

try:
    import requests
    import streamlit as st
    st.set_page_config(page_title="Forest OS", layout="wide", initial_sidebar_state="expanded")
    import json
    import uuid
    from datetime import datetime
    from typing import Dict, List, Union, Optional, Any, Callable
    import graphviz
    from streamlit_cookies_manager import EncryptedCookieManager
    from forest_app.front_end.auth_ui import display_auth_sidebar
    from forest_app.front_end.api_client import call_forest_api
    from forest_app.front_end.onboarding_ui import display_onboarding_input
    from forest_app.core.logging_tracking import setup_global_rotating_error_log

    # Set up global rotating error log for the Streamlit app
    setup_global_rotating_error_log()


    # --- Confirmation Modal Helper ---
    def show_confirmation_modal():
        pending = st.session_state.get("pending_confirmation_action")
        if not pending:
            return
        action = pending.get("action")
        codename = pending.get("codename", "N/A")
        timestamp = pending.get("timestamp", "N/A")
        display_key = pending.get("display_key", "")
        extra_msg = pending.get("extra_msg", "")
        with st.sidebar:
            if action == "reset":
                st.warning("Are you sure you want to reset the environment? This will clear your session and return you to onboarding.")
            else:
                st.info(f"Are you sure you want to {action} this snapshot?")
                st.write(f"**Codename:** {codename}")
                st.write(f"**Timestamp:** {timestamp}")
                if extra_msg:
                    st.write(extra_msg)
            col_c, col_x = st.columns(2)
            with col_c:
                if st.button("✅ Confirm", key="confirm_action_btn"):
                    cb = pending.get("callback")
                    if cb:
                        cb()
                    st.session_state["pending_confirmation_action"] = None
                    
            with col_x:
                if st.button("❌ Cancel", key="cancel_action_btn"):
                    st.session_state["pending_confirmation_action"] = None
                    

    # Assuming constants are defined in a backend config or a separate constants file
    class constants: # Placeholder class if not importing from backend
        ONBOARDING_STATUS_NEEDS_GOAL = "needs_goal"
        ONBOARDING_STATUS_NEEDS_CONTEXT = "needs_context"
        ONBOARDING_STATUS_COMPLETED = "completed"
        MIN_PASSWORD_LENGTH = 8

    # --- Configuration ---
    BACKEND_URL = "http://localhost:8000"
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Constants ---
    # (Consider moving these to state_keys.py later)
    KEY_STATUS_CODE = "status_code"; KEY_ERROR = "error"; KEY_DETAIL = "detail"; KEY_DATA = "data"
    KEY_ACCESS_TOKEN = "access_token"; KEY_ONBOARDING_STATUS = "onboarding_status"
    KEY_USER_INFO_EMAIL = "email"; KEY_USER_INFO_ID = "id"; KEY_SNAPSHOT_ID = "id"
    KEY_SNAPSHOT_UPDATED_AT = "updated_at"; KEY_SNAPSHOT_CODENAME = "codename"; KEY_MESSAGES = "messages"
    KEY_CURRENT_TASK = "current_task"; KEY_HTA_STATE = "hta_state"; KEY_PENDING_CONFIRMATION = "pending_confirmation"
    KEY_MILESTONES = "milestones_achieved"; KEY_TASK_TITLE = "title"; KEY_TASK_DESC = "description"
    KEY_TASK_MAGNITUDE_DESC = "magnitude_description"; KEY_TASK_INTRO_PROMPT = "introspective_prompt"
    KEY_COMMAND_RESPONSE = "arbiter_response"; KEY_COMMAND_OFFERING = "offering"; KEY_COMMAND_MASTERY = "mastery_challenge"
    KEY_ERROR_MESSAGE = "error_message";

    # --- HTA Node Status Constants ---
    STATUS_PENDING = "pending"; STATUS_ACTIVE = "active"; STATUS_COMPLETED = "completed"; STATUS_PRUNED = "pruned"; STATUS_BLOCKED = "blocked";

    # --- HTA Fetching Helper ---
    import uuid
    def sanitize_hta_tree_ids(node):
        """Recursively assign a unique 'id' to any HTA node missing one."""
        if isinstance(node, dict):
            if 'id' not in node or not node['id']:
                node['id'] = str(uuid.uuid4())
            children = node.get('children', [])
            if isinstance(children, list):
                for child in children:
                    sanitize_hta_tree_ids(child)
        return node

    def fetch_hta_state():
        """Fetches HTA state from the backend, handles errors, and updates session state."""
        logger.info("Attempting to fetch HTA state...")
        st.session_state[KEY_ERROR_MESSAGE] = None
        hta_response = call_forest_api( # Uses imported function
            endpoint="/hta/state",
            method="GET",
            backend_url=BACKEND_URL,
            api_token=st.session_state.get("token")
        )
        status_code = hta_response.get(KEY_STATUS_CODE)
        error_msg = hta_response.get(KEY_ERROR)
        hta_data = hta_response.get(KEY_DATA)
        if error_msg:
            log_once_per_session('error',f"Failed fetch HTA: {error_msg} (Status: {status_code})")
            st.session_state[KEY_ERROR_MESSAGE] = f"API Error fetching HTA: {error_msg}"
            st.session_state[KEY_HTA_STATE] = None
        elif status_code == 200:
            if isinstance(hta_data, dict) and 'hta_tree' in hta_data:
                hta_tree_content = hta_data.get('hta_tree')
                if isinstance(hta_tree_content, dict):
                    hta_tree_content = sanitize_hta_tree_ids(hta_tree_content)
                    st.session_state[KEY_HTA_STATE] = hta_tree_content
                    logger.info("Fetched and stored HTA state (IDs sanitized).")
                elif hta_tree_content is None:
                    st.session_state[KEY_HTA_STATE] = None
                    logger.info("Backend indicated no HTA state.")
                else:
                    log_once_per_session('warning',f"HTA 'hta_tree' has unexpected type: {type(hta_tree_content)}")
                    st.session_state[KEY_HTA_STATE] = None
                    st.session_state[KEY_ERROR_MESSAGE] = "Unexpected HTA format."
            else:
                log_once_per_session('warning',f"HTA endpoint gave unexpected structure: {type(hta_data)}")
                st.session_state[KEY_HTA_STATE] = None
                st.session_state[KEY_ERROR_MESSAGE] = "Unexpected HTA structure."
        elif status_code == 404:
            st.session_state[KEY_HTA_STATE] = None
            logger.info("Backend returned 404 for HTA state.")
        else:
            log_once_per_session('error',f"Failed fetch HTA: Status {status_code}. Data: {str(hta_data)[:200]}")
            st.session_state[KEY_ERROR_MESSAGE] = f"Unexpected API status ({status_code}) fetching HTA."
            st.session_state[KEY_HTA_STATE] = None


    # --- HTA Visualization Helpers ---
    STATUS_COLORS = { STATUS_PENDING: "#E0E0E0", STATUS_ACTIVE: "#ADD8E6", STATUS_COMPLETED: "#90EE90", STATUS_PRUNED: "#A9A9A9", STATUS_BLOCKED: "#FFCCCB", "default": "#FFFFFF" }
    def build_hta_dot_string(node_data: Dict[str, Any], dot: graphviz.Digraph, missing_id_flag: set = None):
        if missing_id_flag is None:
            missing_id_flag = set()
        node_id = node_data.get("id")
        if not node_id:
            if "warned" not in missing_id_flag:
                log_once_per_session('warning',"Skip HTA node missing 'id'. (Further missing 'id' nodes will not be logged this render)")
                missing_id_flag.add("warned")
            return
        node_title = node_data.get("title", "Untitled"); node_status = str(node_data.get("status", "default")).lower()
        node_color = STATUS_COLORS.get(node_status, STATUS_COLORS["default"]); node_label = f"{node_title}\n(Status: {node_status.capitalize()})"
        dot.node(name=str(node_id), label=node_label, shape="box", style="filled", fillcolor=node_color)
        children = node_data.get("children", [])
        if isinstance(children, list):
            for child_data in children:
                if isinstance(child_data, dict):
                    child_id = child_data.get("id")
                    if child_id: dot.edge(str(node_id), str(child_id)); build_hta_dot_string(child_data, dot, missing_id_flag)
                else: log_once_per_session('warning',f"Skip non-dict child under node {node_id}.")

    def display_hta_visualization(hta_tree_root: Optional[Dict]):
        if not hta_tree_root or not isinstance(hta_tree_root, dict):
            st.info("🌱 Your skill tree (HTA) is being cultivated..."); return
        try:
            dot = graphviz.Digraph(comment='HTA Skill Tree'); dot.attr(rankdir='TB')
            build_hta_dot_string(hta_tree_root, dot)
            st.graphviz_chart(dot); st.caption("Current Skill Tree Visualization")
        except Exception as e: logger.exception("HTA viz render exception!"); st.error(f"Error generating HTA viz: {e}")


    # --- Completion Confirmation Helper ---
    def handle_completion_confirmation():
        pending_conf = st.session_state.get(KEY_PENDING_CONFIRMATION)
        if not isinstance(pending_conf, dict):
            return
        prompt_text = pending_conf.get("prompt", "Confirm?")
        node_id_to_confirm = pending_conf.get("hta_node_id")
        if not node_id_to_confirm:
            log_once_per_session('error', "Confirm missing 'hta_node_id'.")
            st.error("Confirm prompt missing ID.")
            st.session_state[KEY_PENDING_CONFIRMATION] = None
            return
        st.info(f"**Confirmation Needed:** {prompt_text}")
        col_confirm, col_deny = st.columns(2)
        with col_confirm:
            if st.button("✅ Yes", key=f"confirm_yes_{node_id_to_confirm}"):
                st.session_state[KEY_ERROR_MESSAGE] = None; logger.info(f"User confirm node: {node_id_to_confirm}")
                payload = {"task_id": node_id_to_confirm, "success": True}
                with st.spinner("Confirming..."):
                    response = call_forest_api("/core/complete_task", method="POST", data=payload, backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                if response.get(KEY_ERROR): error_msg = response.get(KEY_ERROR, "Unknown"); log_once_per_session('error',f"API error confirm task {node_id_to_confirm}: {error_msg}"); st.error(f"Confirm Fail: {error_msg}"); st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {error_msg}"
                elif response.get(KEY_STATUS_CODE) == 200:
                    st.success("Confirmed!"); logger.info(f"Success confirm node {node_id_to_confirm}."); st.session_state[KEY_PENDING_CONFIRMATION] = None
                    resp_data = response.get(KEY_DATA, {});
                    if isinstance(resp_data, dict):
                        completion_message = resp_data.get("detail", resp_data.get("message"))
                        if completion_message:
                            if not isinstance(st.session_state.get(KEY_MESSAGES), list): st.session_state[KEY_MESSAGES] = []
                            st.session_state.messages.append({"role": "assistant", "content": str(completion_message)})
                        challenge_data = resp_data.get("result", {}).get(KEY_COMMAND_MASTERY)
                        if isinstance(challenge_data, dict):
                            challenge_content = challenge_data.get("challenge_content", "Consider next steps."); challenge_type = challenge_data.get("challenge_type", "Reflect")
                            logger.info(f"Mastery challenge ({challenge_type}) received.");
                            if not isinstance(st.session_state.get(KEY_MESSAGES), list): st.session_state[KEY_MESSAGES] = []
                            st.session_state.messages.append({"role": "assistant", "content": f"✨ Mastery Challenge ({challenge_type}):\n{challenge_content}"})
                    fetch_hta_state(); 
                else: log_once_per_session('error',f"Unexpected status {response.get(KEY_STATUS_CODE)} confirm task {node_id_to_confirm}."); st.error(f"Unexpected confirm status ({response.get(KEY_STATUS_CODE)})."); st.session_state[KEY_ERROR_MESSAGE] = f"API Error: Confirm status {response.get(KEY_STATUS_CODE)}"
        with col_deny:
            if st.button("❌ No", key=f"confirm_no_{node_id_to_confirm}"):
                st.session_state[KEY_ERROR_MESSAGE] = None; logger.info(f"User denied node: {node_id_to_confirm}")
                st.info("Okay, task not marked complete."); st.session_state[KEY_PENDING_CONFIRMATION] = None
                if not isinstance(st.session_state.get(KEY_MESSAGES), list): st.session_state[KEY_MESSAGES] = []
                st.session_state.messages.append({"role": "assistant", "content": "Okay, let me know when you're ready."}); 


    # --- REMOVED Onboarding Handler Function Definitions ---
    # handle_set_goal and handle_add_context moved to onboarding_ui.py


    # --- Streamlit App Layout ---
    st.title("🌳 Forest OS - Your Growth Companion")

    # --- Initialize Session State ---
    st.session_state.setdefault("authenticated", False); st.session_state.setdefault("token", None)
    st.session_state.setdefault("user_info", None); st.session_state.setdefault(KEY_MESSAGES, [])
    st.session_state.setdefault(KEY_CURRENT_TASK, None); st.session_state.setdefault(KEY_ONBOARDING_STATUS, None)
    st.session_state.setdefault("snapshots", []); st.session_state.setdefault(KEY_ERROR_MESSAGE, None)
    st.session_state.setdefault(KEY_HTA_STATE, None); st.session_state.setdefault(KEY_PENDING_CONFIRMATION, None)
    st.session_state.setdefault(KEY_MILESTONES, [])

    # --- Persistent Login with Query Params (Streamlit 1.32+) ---
    query_params = st.query_params
    if not st.session_state.get("token") and "token" in query_params:
        st.session_state["token"] = query_params["token"]
        st.session_state["authenticated"] = True

    # --- Authentication UI (Sidebar) ---
    with st.sidebar:
        # Call the display function from the auth_ui module
        auth_action_taken = display_auth_sidebar(backend_url=BACKEND_URL)
        # On successful login, set the token in the query params
        if st.session_state.get("authenticated") and st.session_state.get("token"):
            st.query_params["token"] = st.session_state["token"]
        # On logout, remove the token from the query params
        if not st.session_state.get("authenticated"):
            if "token" in st.query_params:
                del st.query_params["token"]

        # --- Snapshot Management (Keep here for now, refactor later) ---
        st.divider()
        if st.session_state.get("authenticated"):
            with st.expander("Snapshot Management", expanded=False):
                st.info("Snapshots allow saving and loading session states (experimental).")
                # Always fetch the latest snapshot list
                if st.button("Refresh Snapshot List") or "snapshots" not in st.session_state:
                    st.session_state[KEY_ERROR_MESSAGE] = None
                    with st.spinner("Fetching snapshot list..."):
                        response = call_forest_api("/snapshots/list", method="GET", backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                        if response.get(KEY_ERROR):
                            st.error(f"Fetch Snap Fail: {response.get(KEY_ERROR)}")
                            st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {response.get(KEY_ERROR)}"
                            st.session_state.snapshots = []
                        elif response.get(KEY_STATUS_CODE) == 200 and isinstance(response.get(KEY_DATA), list):
                            snapshot_list_data = response.get(KEY_DATA, [])
                            valid_snapshots = [item for item in snapshot_list_data if isinstance(item, dict) and KEY_SNAPSHOT_ID in item and KEY_SNAPSHOT_UPDATED_AT in item]
                            try:
                                st.session_state.snapshots = sorted(valid_snapshots, key=lambda x: datetime.fromisoformat(str(x[KEY_SNAPSHOT_UPDATED_AT]).replace('Z', '+00:00')), reverse=True)
                            except Exception as sort_e:
                                st.session_state.snapshots = valid_snapshots
                                st.warning("Snapshots loaded, sort fail.")
                        else:
                            st.session_state.snapshots = []
                            st.info("No snapshots found.")

                # Dropdown for snapshots
                snapshot_options = {}
                for s in st.session_state.snapshots:
                    snap_id = s.get(KEY_SNAPSHOT_ID)
                    if not snap_id: continue
                    codename = s.get(KEY_SNAPSHOT_CODENAME, 'Untitled')
                    updated_at_raw = s.get(KEY_SNAPSHOT_UPDATED_AT)
                    dt_str = 'Date N/A'
                    if updated_at_raw:
                        try:
                            dt_obj = datetime.fromisoformat(str(updated_at_raw).replace('Z', '+00:00'))
                            dt_str = dt_obj.strftime('%Y-%m-%d %H:%M UTC')
                        except (ValueError, TypeError):
                            pass
                    display_key = f"'{codename}' ({dt_str}) - ID: ...{str(snap_id)[-6:]}"
                    snapshot_options[display_key] = snap_id
                selected_snapshot_display = st.selectbox("Select Snapshot:", options=list(snapshot_options.keys()), key="snap_select") if snapshot_options else None
                snapshot_id_selected = snapshot_options.get(selected_snapshot_display) if selected_snapshot_display else None
                # Get selected snapshot info for confirmation
                selected_snapshot = next((s for s in st.session_state.snapshots if s.get(KEY_SNAPSHOT_ID) == snapshot_id_selected), None)
                codename = selected_snapshot.get(KEY_SNAPSHOT_CODENAME, 'Untitled') if selected_snapshot else ''
                timestamp = selected_snapshot.get(KEY_SNAPSHOT_UPDATED_AT, 'N/A') if selected_snapshot else ''
                # --- Action Callbacks ---
                def save_callback():
                    response = call_forest_api("/core/command", method="POST", data={"command": "/save"}, backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                    if response.get(KEY_ERROR):
                        st.error(f"Save Fail: {response.get(KEY_ERROR)}")
                        st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {response.get(KEY_ERROR)}"
                    elif response.get(KEY_STATUS_CODE) == 200:
                        st.success("Snapshot saved!")
                    else:
                        st.error(f"Save Fail: Status {response.get(KEY_STATUS_CODE)}")
                def load_callback():
                    if snapshot_id_selected:
                        load_payload = {"snapshot_id": snapshot_id_selected}
                        # response = call_forest_api("/core/session/load", method="POST", data=load_payload, backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                        if response.get(KEY_ERROR):
                            st.error(f"Load Fail: {response.get(KEY_ERROR)}")
                            st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {response.get(KEY_ERROR)}"
                        elif response.get(KEY_STATUS_CODE) == 200 and isinstance(response.get(KEY_DATA), dict):
                            load_data = response.get(KEY_DATA, {})
                            st.success(load_data.get("message", "Loaded!"))
                            st.session_state[KEY_MESSAGES] = load_data.get(KEY_MESSAGES, []) if isinstance(load_data.get(KEY_MESSAGES), list) else []
                            st.session_state[KEY_CURRENT_TASK] = load_data.get(KEY_CURRENT_TASK) if isinstance(load_data.get(KEY_CURRENT_TASK), dict) else None
                            st.session_state[KEY_MILESTONES] = load_data.get(KEY_MILESTONES, []) if isinstance(load_data.get(KEY_MILESTONES), list) else []
                            st.session_state[KEY_ERROR_MESSAGE] = None
                            st.session_state[KEY_ONBOARDING_STATUS] = constants.ONBOARDING_STATUS_COMPLETED
                            st.session_state[KEY_HTA_STATE] = None
                            st.session_state[KEY_PENDING_CONFIRMATION] = None
                            fetch_hta_state()
                        else:
                            st.error(f"Load Fail: Status {response.get(KEY_STATUS_CODE)}")
                def delete_callback():
                    if snapshot_id_selected:
                        delete_endpoint = f"/snapshots/{snapshot_id_selected}"
                        response = call_forest_api(delete_endpoint, method="DELETE", backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                        if response.get(KEY_ERROR):
                            st.error(f"Delete Fail: {response.get(KEY_ERROR)}")
                            st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {response.get(KEY_ERROR)}"
                        elif response.get(KEY_STATUS_CODE) in [200, 204]:
                            st.success("Deleted.")
                            if isinstance(st.session_state.get("snapshots"), list):
                                st.session_state.snapshots = [s for s in st.session_state.snapshots if s.get(KEY_SNAPSHOT_ID) != snapshot_id_selected]
                        else:
                            st.error(f"Delete Fail: Status {response.get(KEY_STATUS_CODE)}")
                def reset_callback():
                    # Clear all session state and set onboarding
                    for k in list(st.session_state.keys()):
                        if k not in ["authenticated", "token", "user_info"]:
                            del st.session_state[k]
                    st.session_state[KEY_ONBOARDING_STATUS] = constants.ONBOARDING_STATUS_NEEDS_GOAL
                    st.session_state[KEY_MESSAGES] = []
                    st.session_state[KEY_CURRENT_TASK] = None
                    st.session_state[KEY_MILESTONES] = []
                    st.session_state[KEY_ERROR_MESSAGE] = None
                    st.session_state[KEY_HTA_STATE] = None
                    st.session_state[KEY_PENDING_CONFIRMATION] = None
                    st.session_state["batch"] = []
                    st.success("Environment reset. Please start onboarding.")
                # --- Action Buttons with Confirmation ---
                col_save, col_load, col_delete, col_reset = st.columns(4)
                with col_save:
                    if st.button("Save New Snapshot", key="save_snap", help="Save current session state"):
                        st.session_state["pending_confirmation_action"] = {
                            "action": "save",
                            "codename": codename,
                            "timestamp": timestamp,
                            "display_key": selected_snapshot_display,
                            "callback": save_callback,
                        }
                with col_load:
                    if st.button("Load Selected", key="load_snap", help="Load session state"):
                        if snapshot_id_selected:
                            st.session_state["pending_confirmation_action"] = {
                                "action": "load",
                                "codename": codename,
                                "timestamp": timestamp,
                                "display_key": selected_snapshot_display,
                                "callback": load_callback,
                            }
                        else:
                            st.warning("No snapshot selected.")
                with col_delete:
                    if st.button("Delete Selected", type="secondary", key="delete_snap", help="Permanently delete"):
                        if snapshot_id_selected:
                            st.session_state["pending_confirmation_action"] = {
                                "action": "delete",
                                "codename": codename,
                                "timestamp": timestamp,
                                "display_key": selected_snapshot_display,
                                "callback": delete_callback,
                            }
                        else:
                            st.warning("No snapshot selected.")
                with col_reset:
                    if st.button("Reset Environment", key="reset_env", help="Clear all state and return to onboarding"):
                        st.session_state["pending_confirmation_action"] = {
                            "action": "reset",
                            "codename": "reset",
                            "timestamp": "now",
                            "display_key": "reset",
                            "callback": reset_callback,
                        }
                # Show confirmation modal if needed
                show_confirmation_modal()

        # Display global errors
        global_error = st.session_state.get(KEY_ERROR_MESSAGE)
        if global_error:
            st.sidebar.error(f"⚠️ Error: {global_error}")

    # --- Post-Auth Action Handling ---
    if auth_action_taken:
        if st.session_state.get("authenticated"):
            # Fetch the latest snapshot for the user
            response = call_forest_api(
                "/snapshots/list",
                method="GET",
                backend_url=BACKEND_URL,
                api_token=st.session_state.get("token")
            )
            snapshots = response.get("data", []) if response.get("status_code") == 200 else []
            if snapshots:
                # Find the most recent snapshot
                latest = max(snapshots, key=lambda s: s.get("created_at", ""))
                snapshot_id = latest.get("id")
                if snapshot_id:
                    # Load the latest snapshot
                    load_payload = {"snapshot_id": snapshot_id}
                    # response = call_forest_api("/core/session/load", method="POST", data=load_payload, backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                    if response.get(KEY_ERROR):
                        st.error(f"Load Fail: {response.get(KEY_ERROR)}")
                        st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {response.get(KEY_ERROR)}"
                    elif response.get(KEY_STATUS_CODE) == 200 and isinstance(response.get(KEY_DATA), dict):
                        load_data = response.get(KEY_DATA, {})
                        st.success(load_data.get("message", "Loaded!"))
                        st.session_state[KEY_MESSAGES] = load_data.get(KEY_MESSAGES, []) if isinstance(load_data.get(KEY_MESSAGES), list) else []
                        st.session_state[KEY_CURRENT_TASK] = load_data.get(KEY_CURRENT_TASK) if isinstance(load_data.get(KEY_CURRENT_TASK), dict) else None
                        st.session_state[KEY_MILESTONES] = load_data.get(KEY_MILESTONES, []) if isinstance(load_data.get(KEY_MILESTONES), list) else []
                        st.session_state[KEY_ERROR_MESSAGE] = None
                        st.session_state[KEY_ONBOARDING_STATUS] = constants.ONBOARDING_STATUS_COMPLETED
                        st.session_state[KEY_HTA_STATE] = None
                        st.session_state[KEY_PENDING_CONFIRMATION] = None
                        fetch_hta_state()
                    else:
                        st.error(f"Load Fail: Status {response.get(KEY_STATUS_CODE)}")
                # Prevent infinite rerun loop: only rerun if not just rerun
                if not st.session_state.get("_just_reran", False):
                    st.session_state["_just_reran"] = True
                    
                else:
                    if not st.session_state.get("_rerun_error_logged", False):
                        log_once_per_session('error',"Prevented rerun loop at block 450-456.")
                        st.session_state["_rerun_error_logged"] = True
                    st.session_state["_just_reran"] = False

    # --- Main Application Area ---
    if not st.session_state.get("authenticated"):
        st.warning("Please log in or register using the sidebar to begin.")
        st.image("https://placehold.co/800x400/334455/FFFFFF?text=Welcome+to+Forest+OS", caption="Visualize your growth journey")
    else:
        # Check/Fetch User Status if missing
        if st.session_state.get(KEY_ONBOARDING_STATUS) is None and st.session_state.get("token"):
            if not st.session_state.get("_onboarding_status_missing_logged", False):
                log_once_per_session('warning',"Onboarding status missing, refreshing...")
                st.session_state["_onboarding_status_missing_logged"] = True
            with st.spinner("Checking status..."):
                user_details_response = call_forest_api("/users/me", method="GET", backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                if user_details_response.get(KEY_ERROR) or not isinstance(user_details_response.get(KEY_DATA), dict):
                    log_once_per_session('error',"Failed refresh user status."); st.error("Status retrieval failed. Try logout/login."); st.session_state[KEY_ONBOARDING_STATUS] = "error"
                else:
                    user_data = user_details_response[KEY_DATA]; st.session_state.user_info = user_data
                    user_onboarding_status = user_data.get(KEY_ONBOARDING_STATUS); valid_statuses = [constants.ONBOARDING_STATUS_NEEDS_GOAL, constants.ONBOARDING_STATUS_NEEDS_CONTEXT, constants.ONBOARDING_STATUS_COMPLETED]
                    if user_onboarding_status in valid_statuses:
                        st.session_state[KEY_ONBOARDING_STATUS] = user_onboarding_status; logger.info(f"Refreshed status: {st.session_state[KEY_ONBOARDING_STATUS]}")
                        if st.session_state[KEY_ONBOARDING_STATUS] == constants.ONBOARDING_STATUS_COMPLETED and not st.session_state.get(KEY_HTA_STATE): fetch_hta_state()
                        
                    else: log_once_per_session('error',f"Invalid status received: {user_onboarding_status}"); st.error("Invalid status from backend."); st.session_state[KEY_ONBOARDING_STATUS] = "error"

        # Main Content Area Layout
        col_hta, col_chat = st.columns([1, 1])

        # HTA Visualization Column
        with col_hta:
            st.header("Skill Tree (HTA)"); hta_viz_enabled = True; onboarding_status = st.session_state.get(KEY_ONBOARDING_STATUS)
            if onboarding_status == constants.ONBOARDING_STATUS_COMPLETED and hta_viz_enabled:
                if st.button("🔄 Refresh Skill Tree", key="refresh_hta"):
                    with st.spinner("Refreshing..."): fetch_hta_state(); 
                hta_data_to_display = st.session_state.get(KEY_HTA_STATE)
                display_hta_visualization(hta_data_to_display)
            elif not hta_viz_enabled: st.info("Viz disabled.")
            elif onboarding_status in [constants.ONBOARDING_STATUS_NEEDS_GOAL, constants.ONBOARDING_STATUS_NEEDS_CONTEXT]: st.info("Complete onboarding to view Skill Tree.")
            elif onboarding_status == "error": st.warning("Cannot display Skill Tree due to status error.")

        # Chat / Interaction Column
        with col_chat:
            st.header("Conversation & Actions")
            # Display Chat History
            messages_list = st.session_state.get(KEY_MESSAGES, [])
            if isinstance(messages_list, list):
                for message in messages_list:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        with st.chat_message(message["role"]): st.markdown(str(message["content"]))
                    else: log_once_per_session('warning',f"Skip invalid message: {message}")
            elif messages_list: log_once_per_session('error',f"Chat history not list: {type(messages_list)}"); st.error("Chat history corrupt.")

            # Handle Pending Confirmation
            handle_completion_confirmation()

            # Chat Input Logic
            current_status = st.session_state.get(KEY_ONBOARDING_STATUS)
            chat_disabled = st.session_state.get(KEY_PENDING_CONFIRMATION) is not None
            if current_status == constants.ONBOARDING_STATUS_COMPLETED:
                input_placeholder = "Enter reflection, command..."
                if prompt := st.chat_input(input_placeholder, disabled=chat_disabled, key="main_chat"):
                    st.session_state[KEY_ERROR_MESSAGE] = None
                    if not isinstance(st.session_state.get(KEY_MESSAGES), list):
                        st.session_state[KEY_MESSAGES] = []
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown("🌳 Thinking...")
                        api_endpoint = "/core/command"
                        payload = {"command": prompt}
                        response = call_forest_api(api_endpoint, method="POST", data=payload, backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                        assistant_response_content = ""
                        confirmation_details = None  # Default values
                        if response.get(KEY_ERROR):
                            error_msg = response.get(KEY_ERROR, "Command failed")
                            log_once_per_session('warning', f"Error {api_endpoint}: {error_msg} (Status: {response.get(KEY_STATUS_CODE)})")
                            assistant_response_content = f"⚠️ Error: {error_msg}"
                            if response.get(KEY_STATUS_CODE) == 403:
                                assistant_response_content += "\nSession issue? Refreshing status..."
                                st.session_state[KEY_ONBOARDING_STATUS] = None
                            st.session_state[KEY_ERROR_MESSAGE] = assistant_response_content
                        elif response.get(KEY_STATUS_CODE) in [200, 201] and isinstance(response.get(KEY_DATA), dict):
                            resp_data = response.get(KEY_DATA, {})
                            logger.info(f"Success from {api_endpoint}.")
                            assistant_response_content = resp_data.get(KEY_COMMAND_RESPONSE, resp_data.get("message", ""))
                            new_task_data = resp_data.get(KEY_CURRENT_TASK)
                            st.session_state[KEY_CURRENT_TASK] = new_task_data if isinstance(new_task_data, dict) else None
                            action_required = resp_data.get("action_required")
                            confirmation_details = resp_data.get("confirmation_details")
                            milestone_feedback = resp_data.get("milestone_feedback")
                            if milestone_feedback:
                                logger.info(f"Milestone: {milestone_feedback}")
                                if not isinstance(st.session_state.get(KEY_MILESTONES), list):
                                    st.session_state[KEY_MILESTONES] = []
                                st.session_state.milestones_achieved.append(str(milestone_feedback))
                                assistant_response_content += f"\n\n🎉 *Milestone: {milestone_feedback}*"
                            if action_required == "confirm_completion" and isinstance(confirmation_details, dict):
                                logger.info(f"Confirm requested: {confirmation_details.get('hta_node_id')}")
                                st.session_state[KEY_PENDING_CONFIRMATION] = confirmation_details
                                if not assistant_response_content:
                                    assistant_response_content = confirmation_details.get("prompt", "Confirm?")
                            else:
                                if st.session_state.get(KEY_PENDING_CONFIRMATION):
                                    logger.debug("Clearing stale confirm.")
                                    st.session_state[KEY_PENDING_CONFIRMATION] = None
                            if not assistant_response_content:
                                assistant_response_content = "Okay, processed."
                            if new_task_data or milestone_feedback:
                                fetch_hta_state()
                            # --- UPDATE: Set the current batch of tasks from the /core/command response ---
                            st.session_state.batch = resp_data.get("tasks", [])
                        else:
                            status_code = response.get(KEY_STATUS_CODE, "N/A")
                            data_type = type(response.get(KEY_DATA)).__name__
                            log_once_per_session('error', f"Unexpected response {api_endpoint}: Status {status_code}, Type: {data_type}")
                            assistant_response_content = f"Unexpected server response (Status: {status_code})."
                            st.session_state[KEY_ERROR_MESSAGE] = assistant_response_content
                        message_placeholder.markdown(assistant_response_content)
                        if assistant_response_content:
                            if not isinstance(st.session_state.get(KEY_MESSAGES), list):
                                st.session_state[KEY_MESSAGES] = []
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})

            # --- Handle Other/Error States ---
            elif current_status is None: st.info("Checking status...")
            elif current_status == "error": st.error("Status error. Try logout/login.")
            else: st.warning(f"Unknown state ('{current_status}'). Try logout/login.")

    # --- End of App ---

    # --- Inject custom CSS for cleanliness and functionality ---
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
            margin: auto;
        }
        .task-card {
            background: #f8f9fa;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            padding: 1.2rem 1.5rem;
            margin-bottom: 1.2rem;
            border: 1px solid #e0e0e0;
            transition: box-shadow 0.2s;
        }
        .task-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.10);
            border-color: #b0b0b0;
        }
        .stButton>button {
            border-radius: 8px !important;
            padding: 0.5rem 1.2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }
        .stButton>button:disabled {
            background: #e0e0e0 !important;
            color: #888 !important;
            border: 1px solid #ccc !important;
        }
        </style>
    """, unsafe_allow_html=True)

    API_URL = "http://localhost:8000/api"  # Adjust if running elsewhere

    def get_current_batch():
        try:
            response = call_forest_api(
                "/onboarding/current_batch",
                method="GET",
                backend_url=BACKEND_URL,
                api_token=st.session_state.get("token")
            )
            # Accept plain list, dict with 'data' as list, or warn if unexpected structure
            if isinstance(response, list):
                return response
            if isinstance(response, dict):
                data = response.get("data")
                if isinstance(data, list):
                    return data
                # Accept direct 'tasks' key as fallback
                if isinstance(response.get("tasks"), list):
                    return response["tasks"]
            # Warn if structure is not as expected
            st.warning("API /onboarding/current_batch returned unexpected structure. Showing no tasks.")
            return []
        except Exception as e:
            st.error(f"An error occurred while fetching the current batch: {e}")
            return []

    def complete_node(node_id):
        response = call_forest_api(
            "/core/complete_task",
            method="POST",
            data={"task_id": node_id, "success": True},
            backend_url=BACKEND_URL,
            api_token=st.session_state.get("token")
        )
        # Handle 404 or 'Task not found in backlog' gracefully
        if response.get("error"):
            error_msg = str(response["error"])
            if (response.get("status_code") == 404 or "not found in backlog" in error_msg.lower()):
                st.warning("This task could not be found in the backlog. It may have already been completed or removed.")
            else:
                st.error(f"❌ Could not complete task: {error_msg}")
            return []
        result = response.get("data", {}).get("result", {})
        if isinstance(result, dict) and "tasks" in result:
            return result["tasks"]
        return []

    # Only show batch UI if authenticated and onboarding is complete
    if st.session_state.get("authenticated") and st.session_state.get(KEY_ONBOARDING_STATUS) == constants.ONBOARDING_STATUS_COMPLETED:
        if "batch" not in st.session_state:
            st.session_state.batch = get_current_batch()
            st.session_state.completed = set()

        st.write("### Current Batch of Tasks")
        all_completed = True
        any_task_rendered = False
        for task in st.session_state.batch:
            if "hta_node_id" not in task:
                log_once_per_session('warning',f"Skipping task missing 'hta_node_id': {task}")
                continue
            any_task_rendered = True
            node_id = task["hta_node_id"]
            with st.container():
                st.markdown('<div class="task-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{task['title']}**: {task['description']}")
                with col2:
                    if node_id in st.session_state.completed:
                        st.button("Completed", key=f"done_{node_id}", disabled=True)
                    else:
                        if st.button("Mark Complete", key=f"mark_{node_id}"):
                            st.session_state.completed.add(node_id)
                            new_batch = complete_node(node_id)
                            old_ids = set(t["hta_node_id"] for t in st.session_state.batch if "hta_node_id" in t)
                            new_ids = set(t["hta_node_id"] for t in new_batch if "hta_node_id" in t)
                            rerun_needed = False
                            # Reset rerun_attempted flag at the start of a successful run
                            st.session_state["rerun_attempted"] = False
                            # If the new batch is a subset of the old batch, just remove the completed task
                            if new_ids.issubset(old_ids):
                                if any(t.get("hta_node_id") == node_id for t in st.session_state.batch):
                                    st.session_state.batch = [t for t in st.session_state.batch if t.get("hta_node_id") != node_id]
                                    rerun_needed = True
                            else:
                                if st.session_state.batch != new_batch:
                                    st.session_state.batch = new_batch
                                    st.session_state.completed = set()
                                    rerun_needed = True
                            if rerun_needed:
                                if not st.session_state.get("rerun_attempted", False):
                                    st.session_state["rerun_attempted"] = True
                                    
                                else:
                                    st.warning("An error occurred during rerun. Please refresh the page.")
                st.markdown("</div>", unsafe_allow_html=True)
            all_completed = all_completed and (node_id in st.session_state.completed)
        if not any_task_rendered:
            st.info("No actionable tasks available. Please try refreshing or contact support.")

    # Optionally, visualize progress or the tree here

except Exception as e:
    logging.getLogger(__name__).exception("Fatal error in streamlit_app.py startup:")
    raise
