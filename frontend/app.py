import json
import os

import requests
import streamlit as st

st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1000px; padding-top: 2rem; }
    [data-testid="stChatMessage"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("token", "")
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("active_generation", False)
st.session_state.setdefault("active_question", None)

with st.sidebar:
    st.title("Knowledge Assistant")
    api_url = st.text_input("Backend URL", os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
    if not st.session_state.token:
        auth_mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True)
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_password")
        if st.button(auth_mode, use_container_width=True):
            route = "/api/auth/token" if auth_mode == "Sign in" else "/api/auth/register"
            try:
                response = requests.post(
                    f"{api_url}{route}",
                    json={"email": email, "password": password},
                    timeout=20,
                )
                if response.ok:
                    st.session_state.token = response.json()["access_token"]
                    st.rerun()
                st.error(response.json().get("detail", "Authentication failed."))
            except requests.RequestException:
                st.error("Could not reach the chatbot API.")
    else:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        if st.button("Sign out", use_container_width=True):
            st.session_state.token = ""
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.session_state.active_generation = False
            st.session_state.active_question = None
            st.rerun()

        st.divider()
        if st.session_state.active_generation and st.session_state.conversation_id:
            if st.button("Stop generation", use_container_width=True):
                try:
                    stopped = requests.post(
                        f"{api_url}/api/conversations/{st.session_state.conversation_id}/stop",
                        headers=headers,
                        timeout=10,
                    )
                    if stopped.status_code == 202:
                        st.session_state.active_generation = False
                        st.session_state.active_question = None
                        st.rerun()
                    st.warning("No active response could be stopped.")
                except requests.RequestException:
                    st.error("Could not reach the chatbot API to stop generation.")
        if st.button("＋ New conversation", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.session_state.active_generation = False
            st.session_state.active_question = None
            st.rerun()

        st.subheader("Recent conversations")
        try:
            history = requests.get(f"{api_url}/api/conversations", headers=headers, timeout=15)
            if history.ok:
                for conversation in history.json()[:30]:
                    first, second = st.columns([4, 1])
                    if first.button(
                        conversation["title"] or "New conversation",
                        key=f"conversation-{conversation['id']}",
                        use_container_width=True,
                    ):
                        detail = requests.get(
                            f"{api_url}/api/conversations/{conversation['id']}",
                            headers=headers,
                            timeout=15,
                        )
                        if detail.ok:
                            st.session_state.conversation_id = conversation["id"]
                            st.session_state.messages = detail.json()["messages"]
                            st.rerun()
                    if second.button("×", key=f"delete-conversation-{conversation['id']}"):
                        deleted = requests.delete(
                            f"{api_url}/api/conversations/{conversation['id']}",
                            headers=headers,
                            timeout=15,
                        )
                        if deleted.ok:
                            if st.session_state.conversation_id == conversation["id"]:
                                st.session_state.conversation_id = None
                                st.session_state.messages = []
                            st.rerun()
                        st.error("Conversation deletion failed.")
        except requests.RequestException:
            st.warning("Conversation history is unavailable.")

        st.divider()
        st.subheader("Your documents")
        upload = st.file_uploader("Add PDF, DOCX, or TXT", type=["pdf", "docx", "txt", "md"])
        if upload and st.button("Index document", use_container_width=True):
            try:
                response = requests.post(
                    f"{api_url}/api/documents/upload",
                    headers=headers,
                    files={"file": (upload.name, upload.getvalue(), upload.type)},
                    timeout=180,
                )
                if response.ok:
                    st.success(f"Indexed {response.json()['chunks']} chunks.")
                else:
                    st.error(response.json().get("detail", "Document upload failed."))
            except requests.RequestException:
                st.error("Document upload failed because the API is unavailable.")

        try:
            docs = requests.get(f"{api_url}/api/documents", headers=headers, timeout=15)
            if docs.ok:
                for document in docs.json():
                    st.caption(f"{document['filename']} · {document['chunks']} chunks")
                    left, right = st.columns(2)
                    if left.button("Re-index", key=f"reindex-{document['id']}"):
                        result = requests.post(
                            f"{api_url}/api/documents/{document['id']}/reindex",
                            headers=headers,
                            timeout=180,
                        )
                        if result.ok:
                            st.success("Document re-indexed.")
                            st.rerun()
                        st.error("Document re-indexing failed.")
                    if right.button("Delete", key=f"delete-doc-{document['id']}"):
                        result = requests.delete(
                            f"{api_url}/api/documents/{document['id']}",
                            headers=headers,
                            timeout=30,
                        )
                        if result.ok:
                            st.rerun()
                        st.error("Document deletion failed.")
        except requests.RequestException:
            st.warning("Document list is unavailable.")

        with st.expander("Saved memories"):
            memory_text = st.text_area("Save a preference or fact", key="memory-content")
            if st.button("Save memory") and memory_text.strip():
                try:
                    saved = requests.post(
                        f"{api_url}/api/memories",
                        headers=headers,
                        json={"content": memory_text, "category": "preference"},
                        timeout=15,
                    )
                    if saved.ok:
                        st.rerun()
                    st.error("Memory could not be saved.")
                except requests.RequestException:
                    st.error("The chatbot API connection was interrupted.")
            try:
                memories = requests.get(f"{api_url}/api/memories", headers=headers, timeout=15)
                if memories.ok:
                    for memory in memories.json():
                        st.write(memory["content"])
                        if st.button("Delete memory", key=f"delete-memory-{memory['id']}"):
                            requests.delete(
                                f"{api_url}/api/memories/{memory['id']}",
                                headers=headers,
                                timeout=15,
                            )
                            st.rerun()
            except requests.RequestException:
                st.warning("Saved memories are unavailable.")

if not st.session_state.token:
    st.title("Your private knowledge assistant")
    st.write("Sign in or create an account to chat, upload documents, and manage saved memories.")
    st.stop()

st.title("How can I help?")
st.caption("Answers are grounded in your uploaded files when relevant. Sources are shown with each answer.")

for message in st.session_state.messages:
    role = message.get("role", "assistant")
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message.get("content", ""))
        if role != "user":
            with st.expander("Copy response"):
                st.code(message.get("content", ""))
        for citation in message.get("citations", []):
            page_suffix = f" · page {citation['page']}" if citation.get("page") else ""
            with st.expander(f"Source: {citation.get('source', 'document')}{page_suffix}"):
                st.write(citation.get("excerpt", ""))

if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
    if st.button("Regenerate response"):
        prior_question = next(
            (item["content"] for item in reversed(st.session_state.messages[:-1]) if item.get("role") == "user"),
            None,
        )
        if prior_question:
            try:
                regenerated = requests.post(
                    f"{api_url}/api/chat/regenerate",
                    headers={"Authorization": f"Bearer {st.session_state.token}"},
                    json={
                        "question": prior_question,
                        "conversation_id": st.session_state.conversation_id,
                    },
                    timeout=300,
                )
                if regenerated.ok:
                    conversation = requests.get(
                        f"{api_url}/api/conversations/{st.session_state.conversation_id}",
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        timeout=15,
                    )
                    st.session_state.messages = conversation.json()["messages"]
                    st.rerun()
                st.error("The response could not be regenerated.")
            except requests.RequestException:
                st.error("The chatbot API connection was interrupted.")

submitted_question = st.chat_input(
    "Ask a question...",
    disabled=st.session_state.active_generation,
)
if submitted_question:
    with st.chat_message("user"):
        st.markdown(submitted_question)
    st.session_state.messages.append({"role": "user", "content": submitted_question})
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    if not st.session_state.conversation_id:
        try:
            created = requests.post(f"{api_url}/api/conversations", headers=headers, timeout=15)
            created.raise_for_status()
            st.session_state.conversation_id = created.json()["id"]
        except requests.RequestException:
            st.error("Could not start a conversation.")
            st.stop()
    st.session_state.active_question = submitted_question
    st.session_state.active_generation = True
    st.rerun()

if st.session_state.active_generation and st.session_state.active_question:
    question = st.session_state.active_question
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    body = {"question": question, "conversation_id": st.session_state.conversation_id}
    answer_area = st.empty()
    answer_area.markdown("_Thinking..._")
    answer = ""
    citations = []
    stopped = False
    try:
        with requests.post(
            f"{api_url}/api/chat/stream",
            headers=headers,
            json=body,
            stream=True,
            timeout=(15, 300),
        ) as response:
            if not response.ok:
                st.error(response.json().get("detail", "The request failed."))
            else:
                event_name = "message"
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        event_name = "message"
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    if event_name == "conversation":
                        st.session_state.conversation_id = data["id"]
                    elif event_name == "citations":
                        citations = data
                    elif event_name == "error":
                        st.error(data.get("detail", "The assistant could not complete this request."))
                    elif event_name == "stopped":
                        stopped = True
                    elif event_name == "message" and "token" in data:
                        answer += data["token"]
                        answer_area.markdown(answer + "▌")
        st.session_state.active_generation = False
        st.session_state.active_question = None
        if answer and not stopped:
            answer_area.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "citations": citations}
            )
            for citation in citations:
                page_suffix = f" · page {citation['page']}" if citation.get("page") else ""
                with st.expander(f"Source: {citation.get('source', 'document')}{page_suffix}"):
                    st.write(citation.get("excerpt", ""))
            with st.expander("Copy response"):
                st.code(answer)
        elif stopped:
            st.info("Generation stopped.")
    except requests.RequestException:
        st.session_state.active_generation = False
        st.session_state.active_question = None
        st.error("The chatbot API connection was interrupted.")
