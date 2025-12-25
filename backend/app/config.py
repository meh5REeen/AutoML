import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(BASE_DIR, "static", "sessions")

# Ensure the sessions directory exists
os.makedirs(SESSION_DIR, exist_ok=True)
DEBUG = True