import uuid
import os
from app.config import SESSION_DIR

def create_session():
    session_id = str(uuid.uuid4())
    session_path = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(os.path.join(session_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "models"), exist_ok=True)
    return session_id


def get_session_path(session_id: str):
    return os.path.join(SESSION_DIR, session_id)


def clear_session(session_id: str):
    import shutil
    shutil.rmtree(get_session_path(session_id), ignore_errors=True)
