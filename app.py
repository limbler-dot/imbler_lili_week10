import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


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


def ensure_conversations() -> List[Dict[str, Any]]:
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = []
    return st.session_state["conversations"]


def ensure_active_conversation_index() -> int:
    if "active_conversation_index" not in st.session_state:
        st.session_state["active_conversation_index"] = 0
    return int(st.session_state["active_conversation_index"])


def create_conversation() -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"title": timestamp, "messages": []}

def make_title_from_response(text: str, max_len: int = 50) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "Conversation"
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len].rstrip()}..."


st.set_page_config(page_title="My AI Chat", layout="wide", page_icon=":robot_face:")

st.title("Week 10 Chatbot")
st.caption("Chat with your HF inference endpoint and review results.")

conversations = ensure_conversations()
active_index = ensure_active_conversation_index()

with st.sidebar:
    st.header("API Setup")
    st.code(HF_CHAT_COMPLETIONS_URL)
    st.caption(f"Model: `{HF_MODEL_ID}`")
    st.markdown("---")
    st.subheader("Conversations")
    if st.button("New conversation"):
        conversations.append(create_conversation())
        active_index = len(conversations) - 1
        st.session_state["active_conversation_index"] = active_index

    if conversations:
        titles = [c["title"] for c in conversations]
        selected = st.radio(
            "Select a conversation",
            options=list(range(len(titles))),
            format_func=lambda i: titles[i],
            index=min(active_index, len(titles) - 1),
            label_visibility="collapsed",
        )
        active_index = int(selected)
        st.session_state["active_conversation_index"] = active_index
    else:
        st.caption("No conversations yet.")

user_input = st.chat_input("Type your message")

if user_input:
    if not conversations:
        conversations.append(create_conversation())
        active_index = 0
        st.session_state["active_conversation_index"] = active_index

    current_conversation = conversations[active_index]
    current_conversation["messages"].append({"role": "user", "content": user_input})

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
        current_conversation["title"] = make_title_from_response(f"Error: {error_message}")
    else:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": HF_MODEL_ID,
            "messages": current_conversation["messages"],
        }
        with st.spinner("Contacting endpoint..."):
            try:
                response = requests.post(
                    HF_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=60
                )
            except requests.RequestException as exc:
                error_message = f"Request failed: {exc}"
                st.error(error_message)
                current_conversation["messages"].append(
                    {"role": "assistant", "content": f"Error: {error_message}"}
                )
                current_conversation["title"] = make_title_from_response(f"Error: {error_message}")
            else:
                if response.status_code >= 400:
                    error_message = f"HTTP {response.status_code}: {response.text}"
                    st.error(error_message)
                    current_conversation["messages"].append(
                        {"role": "assistant", "content": f"Error: {error_message}"}
                    )
                    current_conversation["title"] = make_title_from_response(f"Error: {error_message}")
                else:
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        error_message = "Response was not valid JSON."
                        st.error(error_message)
                        st.code(response.text)
                        current_conversation["messages"].append(
                            {"role": "assistant", "content": f"Error: {error_message}"}
                        )
                        current_conversation["title"] = make_title_from_response(f"Error: {error_message}")
                    else:
                        generated = extract_generated_text(data)
                        if generated is None:
                            st.warning("Unrecognized response shape. Showing raw JSON.")
                            st.code(json.dumps(data, indent=2))
                            generated = json.dumps(data)
                        current_conversation["messages"].append(
                            {"role": "assistant", "content": generated}
                        )
                        current_conversation["title"] = make_title_from_response(generated)

if conversations:
    current_conversation = conversations[active_index]
    st.subheader("Current Conversation")
    for message in current_conversation["messages"]:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.write(content)
