# front_end/api_client.py
import requests
import json
import logging
from typing import Dict, List, Union, Optional, Any

# Define constants used within this module
# (Consider moving to a central constants/state_keys.py later)
KEY_STATUS_CODE = "status_code"
KEY_ERROR = "error"
KEY_DETAIL = "detail"
KEY_DATA = "data"

# Get logger for this module
logger = logging.getLogger(__name__)

def call_forest_api(
    endpoint: str,
    backend_url: str, # Added parameter
    api_token: Optional[str], # Added parameter
    method: str = "POST",
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Helper function to call the backend API. Returns a consistent dictionary format.
    Modified to remove direct dependency on Streamlit session state.

    Args:
        endpoint (str): API endpoint path (e.g., "/users/me").
        backend_url (str): The base URL of the backend API.
        api_token (Optional[str]): The authentication token (if available).
        method (str): HTTP method ("GET", "POST", "DELETE", etc.). Defaults to "POST".
        data (dict, optional): Payload for POST/PUT requests. Sent as JSON unless endpoint is /auth/token.
        params (dict, optional): URL parameters for GET requests.

    Returns:
        Dict containing:
        {'status_code': int, 'data': Optional[Union[dict, list]], 'error': Optional[str]}
        'data' holds the JSON response body if successful and content exists.
        'error' holds an error message string if the request failed or response was problematic.
    """
    headers = {}
    # Use the passed api_token argument
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # Use the passed backend_url argument
    url = f"{backend_url}{endpoint}"
    response = None
    # Default return structure indicates an internal client error before request attempt
    result = {"status_code": 500, KEY_DATA: None, "KEY_ERROR": "API call initialization error"} # Fixed syntax

    logger.debug(f"Calling API: {method.upper()} {url}")
    # Log payload safely (mask password for token endpoint)
    log_data_repr = "N/A"
    if data:
        is_token_endpoint = endpoint == "/auth/token"
        # For token endpoint, data is form data; otherwise, it's JSON
        payload_type = 'Form' if is_token_endpoint else 'JSON'
        # Mask password in logs
        log_data = {k: v for k, v in data.items() if k != 'password'} if is_token_endpoint else data
        try:
            # Attempt to dump JSON data for logging, fallback to string representation
            log_data_repr = json.dumps(log_data) if not is_token_endpoint else str(log_data)
        except TypeError:
            log_data_repr = str(log_data) # Fallback if data isn't JSON serializable
        logger.debug(f"Payload ({payload_type}): {log_data_repr[:500]}{'...' if len(log_data_repr)>500 else ''}")
    if params:
        logger.debug(f"Params: {params}")

    try:
        # Execute the request based on the method
        method_upper = method.upper()
        if method_upper == "POST":
            if endpoint == "/auth/token": # Special handling for form data
                response = requests.post(url, data=data, headers=headers, params=params, timeout=60)
            else: # Default to JSON payload for other POSTs
                response = requests.post(url, json=data, headers=headers, params=params, timeout=60)
        elif method_upper == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method_upper == "DELETE":
            response = requests.delete(url, headers=headers, params=params, timeout=30)
        # Add PUT, PATCH etc. if needed
        else:
            logger.error(f"Unsupported HTTP method requested: {method}")
            # Syntax fix: Use quoted key
            result = {"status_code": 405, KEY_DATA: None, "KEY_ERROR": f"Unsupported HTTP method: {method}"}
            return result # Return immediately for unsupported method

        # Store status code immediately after getting response
        result["status_code"] = response.status_code
        logger.debug(f"API Raw Response Status: {response.status_code}")

        # --- Handle Non-Success Status Codes (>= 400) ---
        if not response.ok:
            error_detail = f"HTTP Error {response.status_code}"
            try:
                # Try to parse JSON error response (FastAPI often returns {'detail': ...})
                error_json = response.json()
                # Prioritize 'detail', then 'error', then the raw text
                error_detail = error_json.get(KEY_DETAIL, error_json.get(KEY_ERROR, response.text or f"HTTP Error {response.status_code}"))
                logger.warning(f"HTTP Error {response.status_code} calling {url}. Detail: {error_detail}")
            except json.JSONDecodeError:
                # If response is not JSON, use the raw text
                error_detail = response.text or f"HTTP Error {response.status_code} (non-JSON body)"
                logger.warning(f"HTTP Error {response.status_code} calling {url}. Response Text: {error_detail[:500]}")
            result[KEY_ERROR] = str(error_detail) # Ensure error message is a string
            result[KEY_DATA] = None # Ensure data is None on error
            return result # Return immediately on HTTP error

        # --- Handle Success Cases (2xx) ---
        # Handle 204 No Content specifically
        if response.status_code == 204:
            logger.debug(f"API Response: 204 No Content for {url}")
            result[KEY_DATA] = None # Explicitly set data to None
            result[KEY_ERROR] = None # Explicitly set error to None
        # Handle other 2xx responses that might have an empty body
        elif not response.content:
             logger.warning(f"API Response {response.status_code} with empty body for {url}")
             result[KEY_DATA] = None # Explicitly set data to None
             result[KEY_ERROR] = None # Explicitly set error to None
        # Handle 2xx responses with content - attempt JSON parsing
        else:
            try:
                response_json = response.json()
                # Special case: /snapshots/list returns a list directly
                if endpoint == "/snapshots/list" and isinstance(response_json, list):
                    result[KEY_DATA] = response_json
                # Normal case: Expect a dictionary
                elif isinstance(response_json, dict):
                    result[KEY_DATA] = response_json
                # Handle unexpected JSON types (e.g., just a string or number)
                else:
                    logger.warning(f"API Success Response ({response.status_code}) for {url} was JSON but not dict/list: {type(response_json)}")
                    result[KEY_DATA] = response_json # Store it anyway, let caller handle
                result[KEY_ERROR] = None # Clear error on successful parse
                logger.debug(f"API Success Response Data: {str(result[KEY_DATA])[:500]}{'...' if len(str(result[KEY_DATA]))>500 else ''}")
            except json.JSONDecodeError:
                # If JSON parsing fails even on a 2xx response
                logger.error(f"Failed to decode JSON from SUCCESSFUL ({response.status_code}) response from {url}. Response text: {response.text[:500]}{'...' if len(response.text)>500 else ''}")
                result[KEY_DATA] = None
                result[KEY_ERROR] = "Failed to decode JSON response from server, although status was OK."
                # Keep the original success status code, but add the error message

    # --- Handle Network/Request Errors ---
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection Error calling {url}: {conn_err}")
        # Syntax fix: Use quoted key
        result = {"status_code": 503, KEY_DATA: None, "KEY_ERROR": f"Connection error: Could not connect to backend at {backend_url}."}
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout Error calling {url}: {timeout_err}")
        # Syntax fix: Use quoted key
        result = {"status_code": 504, KEY_DATA: None, "KEY_ERROR": f"Timeout error: The request to the backend timed out."}
    except requests.exceptions.RequestException as req_err: # Catch other requests library errors
        logger.error(f"Request Exception calling {url}: {req_err}")
        # Syntax fix: Use quoted key
        result = {"status_code": 500, KEY_DATA: None, "KEY_ERROR": f"Network request error: {req_err}"}
    except Exception as e: # Catch-all for unexpected issues within this function
        logger.exception(f"Unexpected error in call_forest_api for {url}: {e}")
        # Syntax fix: Use quoted key
        result = {"status_code": 500, KEY_DATA: None, "KEY_ERROR": f"An unexpected client-side error occurred while making the API call: {type(e).__name__}"}

    return result
