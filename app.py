import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

def build_endpoint_url(base_url: str, model_id: str) -> str:
    base = base_url.rstrip("/")
    model = model_id.strip()
    return f"{base}/{model}" if model else base


def extract_generated_text(payload: Any) -> Optional[str]:
    # Common HF text-generation responses:
    # 1) [{"generated_text": "..."}]
    # 2) {"generated_text": "..."}
    # 3) {"generated_texts": ["..."]} or {"generated_texts": [{"generated_text": "..."}]}
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"])
    if isinstance(payload, dict):
        if "generated_text" in payload:
            return str(payload["generated_text"])
        if "generated_texts" in payload:
            gts = payload["generated_texts"]
            if isinstance(gts, list) and gts:
                first = gts[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return str(first["generated_text"])
                return str(first)
    return None


def ensure_history() -> List[Dict[str, str]]:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    return st.session_state["history"]


st.set_page_config(page_title="HF Text Generation", page_icon="🧠")
st.title("Hugging Face Text Generation")
st.caption("Enter a prompt, call your HF inference endpoint, and review results.")

history = ensure_history()

with st.sidebar:
    st.header("Endpoint")
    base_url = st.text_input(
        "Base URL",
        placeholder="https://your-endpoint.region.aws.endpoints.huggingface.cloud",
    )
    model_id = st.text_input("Model ID", placeholder="your-model-id")
    st.markdown("---")
    st.subheader("History")
    if history:
        for item in reversed(history):
            st.write(f"**Prompt:** {item['prompt']}")
            st.write(f"**Response:** {item['response']}")
            st.markdown("---")
    else:
        st.caption("No history yet.")

prompt = st.text_area("Prompt", height=160, placeholder="Write a short story about a robot gardener...")

submit = st.button("Generate")

if submit:
    if not base_url.strip():
        st.error("Please provide a base URL.")
    elif not model_id.strip():
        st.error("Please provide a model ID.")
    elif not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        token = st.secrets.get("HF_TOKEN", "").strip()
        if not token:
            st.error("Missing HF_TOKEN in Streamlit secrets.")
        else:
            url = build_endpoint_url(base_url, model_id)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = {"inputs": prompt}
            with st.spinner("Contacting endpoint..."):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
                else:
                    if response.status_code >= 400:
                        st.error(f"HTTP {response.status_code}: {response.text}")
                    else:
                        try:
                            data = response.json()
                        except json.JSONDecodeError:
                            st.error("Response was not valid JSON.")
                            st.code(response.text)
                        else:
                            generated = extract_generated_text(data)
                            if generated is None:
                                st.warning("Unrecognized response shape. Showing raw JSON.")
                                st.code(json.dumps(data, indent=2))
                                generated = json.dumps(data)
                            st.subheader("Response")
                            st.write(generated)
                            history.append({"prompt": prompt, "response": generated})
