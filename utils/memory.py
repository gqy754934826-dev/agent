import json
import os
import uuid
from datetime import datetime
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

STORAGE_DIR = get_abs_path("storage")


def _ensure_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(STORAGE_DIR, f"{session_id}.json")


def _load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_session(session_id: str, session: dict):
    _ensure_dir()
    path = _session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def create_session(title: str = None) -> dict:
    _ensure_dir()
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = {
        "id": session_id,
        "title": title or f"新对话 {now}",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _save_session(session_id, session)
    return session


def get_session(session_id: str) -> dict | None:
    return _load_session(session_id)


def list_sessions() -> list[dict]:
    _ensure_dir()
    result = []
    for filename in os.listdir(STORAGE_DIR):
        if not filename.endswith(".json"):
            continue
        session_id = filename[:-5]
        session = _load_session(session_id)
        if session:
            result.append({
                "id": session["id"],
                "title": session["title"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "message_count": len(session["messages"]),
            })
    result.sort(key=lambda x: x["updated_at"], reverse=True)
    return result


def add_message(session_id: str, role: str, content: str, role_detected: str = "base"):
    session = _load_session(session_id)
    if not session:
        return
    session["messages"].append({
        "role": role,
        "content": content,
        "role_detected": role_detected,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    session["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if role == "user" and len(session["messages"]) == 1:
        session["title"] = content[:20] + "..." if len(content) > 20 else content
    _save_session(session_id, session)


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    session = _load_session(session_id)
    if not session:
        return []
    return session["messages"][-limit:]


def delete_session(session_id: str) -> bool:
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
