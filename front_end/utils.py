import os

def show_recent_errors(log_path, n=50):
    """Return the last n lines of the error log, newest first."""
    if not os.path.exists(log_path):
        return ["Log file not found."]
    with open(log_path, 'r') as f:
        lines = f.readlines()
    return lines[-n:][::-1]  # Return last n lines, reversed
