from langchain_community.chat_message_histories import ChatMessageHistory
from utils.memory import get_history as _load_history


def get_history(session_id: str, **kwargs) -> ChatMessageHistory:
    messages = _load_history(session_id, limit=20)
    history = ChatMessageHistory()
    for msg in messages:
        if msg["role"] == "user":
            history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            history.add_ai_message(msg["content"])
    return history
