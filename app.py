import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests
import streamlit as st


HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path(__file__).resolve().parent / "chats"
MEMORY_PATH = Path(__file__).resolve().parent / "memory.json"


def extract_generated_text(payload: Any) -> Optional[str]:
    # OpenAI-style chat completions: {"choices":[{"message":{"content":"..."}}]}
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
    # Common HF text-generation responses:
    # 1) [{"generated_text": "..."}]
    # 2) {"generated_text": "..."}
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"])
    if isinstance(payload, dict) and "generated_text" in payload:
        return str(payload["generated_text"])
    return None


def iter_sse_content_lines(response: requests.Response) -> Any:
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        yield data


def extract_delta_content(chunk: Dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                return str(delta.get("content") or "")
            message = first.get("message")
            if isinstance(message, dict):
                return str(message.get("content") or "")
    return ""


def load_memory() -> Dict[str, Any]:
    if not MEMORY_PATH.exists():
        return {"facts": []}
    try:
        raw = MEMORY_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return {"facts": []}
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {"facts": []}
    facts = data.get("facts")
    if not isinstance(facts, list):
        facts = []
    return {"facts": facts}


def save_memory(memory: Dict[str, Any]) -> None:
    payload = {"facts": memory.get("facts", [])}
    MEMORY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_facts(existing: List[str], new_facts: List[str]) -> List[str]:
    normalized = {fact.strip(): True for fact in existing if isinstance(fact, str) and fact.strip()}
    for fact in new_facts:
        if isinstance(fact, str) and fact.strip():
            normalized.setdefault(fact.strip(), True)
    return list(normalized.keys())


def extract_memory_facts(token: str, user_message: str) -> List[str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    system_prompt = (
        "Extract personal traits or preferences mentioned by the user in the message. "
        "Examples: name, preferred language, interests, communication style, favorite topics, "
        "or other useful personal preferences. "
        "Return ONLY valid JSON with a single key 'facts' that maps to a list of short strings. "
        "If there are no traits or preferences, return {\"facts\": []}."
    )
    payload = {
        "model": HF_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    try:
        response = requests.post(
            HF_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=30
        )
    except requests.RequestException:
        return []
    if response.status_code >= 400:
        return []
    try:
        data = response.json()
    except json.JSONDecodeError:
        return []
    content = extract_generated_text(data)
    if not content:
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return []
    facts = parsed.get("facts")
    if not isinstance(facts, list):
        return []
    return [str(f).strip() for f in facts if isinstance(f, str) and f.strip()]


def ensure_conversations() -> List[Dict[str, Any]]:
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = load_conversations()
    return st.session_state["conversations"]


def ensure_active_conversation_index() -> int:
    if "active_conversation_index" not in st.session_state:
        st.session_state["active_conversation_index"] = 0
    return int(st.session_state["active_conversation_index"])


def create_conversation() -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": uuid4().hex,
        "title": "New conversation",
        "timestamp": timestamp,
        "messages": [],
    }

def make_title_from_response(text: str, max_len: int = 50) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "Conversation"
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len].rstrip()}..."


def load_conversations() -> List[Dict[str, Any]]:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    conversations: List[Dict[str, Any]] = []
    for path in sorted(CHATS_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        chat_id = str(data.get("id") or path.stem)
        title = str(data.get("title") or "New conversation")
        timestamp = str(
            data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        messages = data.get("messages")
        if not isinstance(messages, list):
            messages = []
        conversations.append(
            {"id": chat_id, "title": title, "timestamp": timestamp, "messages": messages}
        )
    return conversations


def save_conversation(conversation: Dict[str, Any]) -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chat_id = conversation.get("id") or uuid4().hex
    conversation["id"] = chat_id
    path = CHATS_DIR / f"{chat_id}.json"
    payload = {
        "id": conversation.get("id"),
        "title": conversation.get("title"),
        "timestamp": conversation.get("timestamp"),
        "messages": conversation.get("messages", []),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def delete_conversation_file(chat_id: str) -> None:
    if not chat_id:
        return
    path = CHATS_DIR / f"{chat_id}.json"
    try:
        path.unlink()
    except FileNotFoundError:
        pass


st.set_page_config(page_title="My AI Chat", layout="wide", page_icon=":robot_face:")

st.markdown(
    """
    <style>
    section.main div[data-testid="stContainer"] {
        max-height: calc(100vh - 280px);
        overflow-y: auto;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #fdf07e;
        color: #111111;
        border: 1px solid #e6d85a;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #f5e86b;
        color: #111111;
        border: 1px solid #e0d254;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Week 10 Chatbot")
st.caption("Chat with your HF inference endpoint and review results.")

conversations = ensure_conversations()
active_index = ensure_active_conversation_index()
memory = load_memory()

with st.sidebar:
    st.subheader("Conversations")
    if st.button("New Chat"):
        new_chat = create_conversation()
        conversations.append(new_chat)
        save_conversation(new_chat)
        active_index = len(conversations) - 1
        st.session_state["active_conversation_index"] = active_index

    if conversations:
        titles = [c["title"] for c in conversations]
        sidebar_list = st.container(height=380, border=True)
        with sidebar_list:
            delete_index = None
            for i, title in enumerate(titles):
                label = f"{title}".strip()
                row_container = st.container(border=(i == active_index))
                with row_container:
                    cols = st.columns([0.85, 0.15])
                    with cols[0]:
                        button_type = "primary" if i == active_index else "secondary"
                        if st.button(label, key=f"select_chat_{i}", type=button_type):
                            active_index = i
                            st.session_state["active_conversation_index"] = active_index
                    with cols[1]:
                        if st.button("✕", key=f"delete_chat_{i}", type="secondary"):
                            delete_index = i
                            break

            if delete_index is not None:
                removed = conversations.pop(delete_index)
                delete_conversation_file(str(removed.get("id", "")))
                if not conversations:
                    active_index = 0
                elif delete_index == active_index:
                    active_index = min(delete_index, len(conversations) - 1)
                elif delete_index < active_index:
                    active_index -= 1
                st.session_state["active_conversation_index"] = active_index
    else:
        st.caption("No conversations yet.")

user_input = st.chat_input("Type your message")

if user_input:
    if not conversations:
        new_chat = create_conversation()
        conversations.append(new_chat)
        save_conversation(new_chat)
        active_index = 0
        st.session_state["active_conversation_index"] = active_index

    current_conversation = conversations[active_index]
    if current_conversation["title"] == "New conversation":
        current_conversation["title"] = make_title_from_response(user_input)
    save_conversation(current_conversation)
    current_conversation["messages"].append({"role": "user", "content": user_input})
    save_conversation(current_conversation)

    try:
        token = str(st.secrets["HF_TOKEN"]).strip()
    except KeyError:
        token = ""

    if not token:
        error_message = "Missing HF_TOKEN in Streamlit secrets."
        st.error(error_message)
        current_conversation["messages"].append(
            {"role": "assistant", "content": f"Error: {error_message}"}
        )
        save_conversation(current_conversation)
    else:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        messages = current_conversation["messages"]
        if memory.get("facts"):
            system_memory = "Known user preferences: " + "; ".join(memory["facts"])
            messages = [{"role": "system", "content": system_memory}] + messages

        payload = {
            "model": HF_MODEL_ID,
            "messages": messages,
            "stream": True,
        }
        with st.spinner("Contacting endpoint..."):
            try:
                response = requests.post(
                    HF_CHAT_COMPLETIONS_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                    stream=True,
                )
            except requests.RequestException as exc:
                error_message = f"Request failed: {exc}"
                st.error(error_message)
                current_conversation["messages"].append(
                    {"role": "assistant", "content": f"Error: {error_message}"}
                )
                save_conversation(current_conversation)
            else:
                if response.status_code >= 400:
                    error_message = f"HTTP {response.status_code}: {response.text}"
                    st.error(error_message)
                    current_conversation["messages"].append(
                        {"role": "assistant", "content": f"Error: {error_message}"}
                    )
                    save_conversation(current_conversation)
                else:
                    assistant_text = ""
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        for data_line in iter_sse_content_lines(response):
                            try:
                                chunk = json.loads(data_line)
                            except json.JSONDecodeError:
                                continue
                            delta = extract_delta_content(chunk)
                            if delta:
                                assistant_text += delta
                                placeholder.write(assistant_text)
                                time.sleep(0.02)
                        if not assistant_text:
                            st.warning("No streamed content received.")
                    current_conversation["messages"].append(
                        {"role": "assistant", "content": assistant_text or "No response."}
                    )
                    save_conversation(current_conversation)
                    if token and user_input:
                        new_facts = extract_memory_facts(token, user_input)
                        if new_facts:
                            memory["facts"] = merge_facts(memory.get("facts", []), new_facts)
                            save_memory(memory)

if conversations:
    current_conversation = conversations[active_index]
    st.subheader("Current Conversation")
    history_container = st.container(border=True)
    with history_container:
        for message in current_conversation["messages"]:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            with st.chat_message(role):
                st.write(content)
