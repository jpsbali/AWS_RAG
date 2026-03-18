"""
AWS-Native RAG Service — Streamlit Frontend.

Provides three pages via sidebar navigation:
  1. Chat Interface — query the RAG service with streamed responses
  2. Document Upload — upload files for ingestion
  3. Document Browser — view uploaded documents and processing status

Authentication is handled via Cognito hosted UI (OAuth 2.0 / OIDC).
"""

from __future__ import annotations

import base64
import json
import os
import time
from urllib.parse import urlencode

import requests
import streamlit as st
import websocket

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_ENDPOINT: str = os.environ.get("API_ENDPOINT", "http://localhost:8000")
WS_ENDPOINT: str = os.environ.get("WS_ENDPOINT", "ws://localhost:8000/ws")
COGNITO_DOMAIN: str = os.environ.get("COGNITO_DOMAIN", "")
COGNITO_CLIENT_ID: str = os.environ.get("COGNITO_CLIENT_ID", "")
COGNITO_REDIRECT_URI: str = os.environ.get("COGNITO_REDIRECT_URI", "http://localhost:8501")

SUPPORTED_FILE_TYPES: list[str] = ["pdf", "png", "jpeg", "tiff", "docx", "txt", "csv", "html"]
SUPPORTED_MIME_MAP: dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "tiff": "image/tiff",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "csv": "text/csv",
    "html": "text/html",
}

POLL_INTERVAL_SECONDS: int = 3
MAX_POLL_ATTEMPTS: int = 40  # ~120 s total


# ---------------------------------------------------------------------------
# Helpers — authentication
# ---------------------------------------------------------------------------

def _cognito_login_url() -> str:
    """Build the Cognito hosted-UI authorization URL."""
    params = {
        "client_id": COGNITO_CLIENT_ID,
        "response_type": "code",
        "scope": "openid profile email",
        "redirect_uri": COGNITO_REDIRECT_URI,
    }
    return f"https://{COGNITO_DOMAIN}/login?{urlencode(params)}"


def _exchange_code_for_tokens(code: str) -> dict | None:
    """Exchange an authorization code for JWT tokens via the Cognito token endpoint."""
    token_url = f"https://{COGNITO_DOMAIN}/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": COGNITO_CLIENT_ID,
        "code": code,
        "redirect_uri": COGNITO_REDIRECT_URI,
    }
    try:
        resp = requests.post(token_url, data=data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def _auth_headers() -> dict[str, str]:
    """Return Authorization header with the current Bearer token."""
    token = st.session_state.get("id_token", "")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _handle_auth_callback() -> None:
    """Check query params for an authorization code and exchange it."""
    params = st.query_params
    code = params.get("code")
    if code and "id_token" not in st.session_state:
        tokens = _exchange_code_for_tokens(code)
        if tokens:
            st.session_state["id_token"] = tokens.get("id_token", "")
            st.session_state["access_token"] = tokens.get("access_token", "")
            st.session_state["refresh_token"] = tokens.get("refresh_token", "")


def _is_authenticated() -> bool:
    return bool(st.session_state.get("id_token"))


# ---------------------------------------------------------------------------
# Helpers — API calls
# ---------------------------------------------------------------------------

def _api_post(path: str, payload: dict) -> dict:
    """POST to the REST API and return the JSON response."""
    url = f"{API_ENDPOINT.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.post(url, json=payload, headers=_auth_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _api_get(path: str) -> dict:
    """GET from the REST API and return the JSON response."""
    url = f"{API_ENDPOINT.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.get(url, headers=_auth_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _stream_query_ws(query: str) -> str:
    """Send a query over WebSocket and yield streamed tokens, returning the full answer."""
    token = st.session_state.get("id_token", "")
    ws_url = f"{WS_ENDPOINT}?token={token}"
    full_response = ""
    try:
        ws = websocket.create_connection(ws_url, timeout=30)
        ws.send(json.dumps({"action": "query", "query": query}))
        while True:
            result = ws.recv()
            if not result:
                break
            data = json.loads(result)
            if data.get("type") == "token":
                full_response += data.get("content", "")
            elif data.get("type") == "done":
                break
            elif data.get("type") == "error":
                full_response = f"Error: {data.get('message', 'Unknown error')}"
                break
        ws.close()
    except (websocket.WebSocketException, ConnectionError, OSError):
        # Fall back to REST if WebSocket is unavailable
        try:
            resp = _api_post("query", {"query": query, "stream": False})
            full_response = resp.get("answer", "No answer received.")
        except requests.RequestException as exc:
            full_response = f"Error: {exc}"
    return full_response


# ---------------------------------------------------------------------------
# Page — Chat Interface
# ---------------------------------------------------------------------------

def _page_chat() -> None:
    st.header("💬 Chat Interface")

    # Initialise conversation history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display conversation history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        doc_id = src.get("document_id", "unknown")
                        snippet = src.get("content_snippet", "")
                        score = src.get("score", 0)
                        page = src.get("page_number")
                        page_info = f" (page {page})" if page else ""
                        st.markdown(
                            f"- **{doc_id}**{page_info} — score {score:.3f}\n  > {snippet[:200]}"
                        )

    # Query input
    user_query = st.chat_input("Ask a question about your documents…")
    if user_query:
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Try REST first for structured response with sources
                sources: list[dict] = []
                try:
                    resp = _api_post("query", {"query": user_query, "stream": False})
                    answer = resp.get("answer", "No answer received.")
                    sources = resp.get("sources", [])
                except requests.RequestException:
                    # Fall back to WebSocket streaming
                    answer = _stream_query_ws(user_query)

                st.markdown(answer)

                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            doc_id = src.get("document_id", "unknown")
                            snippet = src.get("content_snippet", "")
                            score = src.get("score", 0)
                            page = src.get("page_number")
                            page_info = f" (page {page})" if page else ""
                            st.markdown(
                                f"- **{doc_id}**{page_info} — score {score:.3f}\n  > {snippet[:200]}"
                            )

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


# ---------------------------------------------------------------------------
# Page — Document Upload
# ---------------------------------------------------------------------------

def _page_upload() -> None:
    st.header("📤 Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=SUPPORTED_FILE_TYPES,
        help=f"Supported formats: {', '.join(t.upper() for t in SUPPORTED_FILE_TYPES)}",
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        content_type = SUPPORTED_MIME_MAP.get(file_ext, "application/octet-stream")

        st.info(f"**File:** {file_name}  \n**Type:** {content_type}  \n**Size:** {uploaded_file.size:,} bytes")

        if st.button("Upload & Process", type="primary"):
            # Read file content and base64-encode
            file_bytes = uploaded_file.read()
            file_b64 = base64.b64encode(file_bytes).decode("utf-8")

            progress_bar = st.progress(0, text="Uploading…")

            try:
                # POST /ingest
                payload = {
                    "file_name": file_name,
                    "content_type": content_type,
                    "file_content": file_b64,
                }
                progress_bar.progress(20, text="Sending to API…")
                resp = _api_post("ingest", payload)
                document_id = resp.get("document_id", "")
                progress_bar.progress(40, text=f"Uploaded. Document ID: {document_id}")

                if not document_id:
                    st.error("Upload succeeded but no document ID was returned.")
                    return

                # Poll GET /document/{id} for processing status
                status_placeholder = st.empty()
                for attempt in range(MAX_POLL_ATTEMPTS):
                    time.sleep(POLL_INTERVAL_SECONDS)
                    pct = min(40 + int(55 * (attempt + 1) / MAX_POLL_ATTEMPTS), 95)
                    try:
                        doc_resp = _api_get(f"document/{document_id}")
                        doc_status = doc_resp.get("status", "unknown")
                        progress_bar.progress(pct, text=f"Processing… Status: {doc_status}")
                        status_placeholder.info(f"Document **{document_id}** — status: **{doc_status}**")

                        if doc_status in ("completed", "indexed", "success"):
                            progress_bar.progress(100, text="Processing complete!")
                            st.success(f"Document **{file_name}** processed successfully.")
                            return
                        if doc_status in ("failed", "error"):
                            progress_bar.progress(100, text="Processing failed.")
                            st.error(
                                f"Document processing failed: {doc_resp.get('error', 'Unknown error')}"
                            )
                            return
                    except requests.RequestException:
                        status_placeholder.warning("Could not fetch status, retrying…")

                st.warning("Processing is still in progress. Check the Document Browser for updates.")

            except requests.RequestException as exc:
                progress_bar.progress(100, text="Upload failed.")
                st.error(f"Upload failed: {exc}")


# ---------------------------------------------------------------------------
# Page — Document Browser
# ---------------------------------------------------------------------------

def _page_browser() -> None:
    st.header("📁 Document Browser")

    # Allow user to enter a document ID to look up
    doc_id_input = st.text_input("Enter a Document ID to view details", "")

    if doc_id_input:
        try:
            doc = _api_get(f"document/{doc_id_input}")
            st.subheader(f"Document: {doc.get('file_name', doc_id_input)}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", doc.get("status", "unknown"))
                st.write(f"**Document ID:** {doc.get('document_id', doc_id_input)}")
                st.write(f"**File Name:** {doc.get('file_name', 'N/A')}")
                st.write(f"**Content Type:** {doc.get('content_type', 'N/A')}")
            with col2:
                st.write(f"**Created:** {doc.get('created_at', 'N/A')}")
                st.write(f"**Updated:** {doc.get('updated_at', 'N/A')}")
                st.write(f"**Size:** {doc.get('file_size', 'N/A')}")

            # Show metadata if available
            metadata = doc.get("metadata")
            if metadata:
                with st.expander("📋 Metadata"):
                    st.json(metadata)

            # Show processing error if failed
            if doc.get("status") in ("failed", "error"):
                st.error(f"Processing error: {doc.get('error', 'Unknown error')}")

        except requests.RequestException as exc:
            st.error(f"Could not fetch document: {exc}")

    # System health check
    st.divider()
    st.subheader("System Status")
    col_health, col_metrics = st.columns(2)

    with col_health:
        if st.button("Check Health"):
            try:
                health = _api_get("health")
                st.success("System is healthy")
                st.json(health)
            except requests.RequestException as exc:
                st.error(f"Health check failed: {exc}")

    with col_metrics:
        if st.button("View Metrics"):
            try:
                metrics = _api_get("metrics")
                st.json(metrics)
            except requests.RequestException as exc:
                st.error(f"Could not fetch metrics: {exc}")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="RAG Service",
        page_icon="🔍",
        layout="wide",
    )

    # Handle OAuth callback
    _handle_auth_callback()

    # Sidebar navigation
    st.sidebar.title("🔍 RAG Service")

    # Authentication gate
    if COGNITO_DOMAIN and COGNITO_CLIENT_ID:
        if not _is_authenticated():
            st.sidebar.warning("Please log in to continue.")
            login_url = _cognito_login_url()
            st.sidebar.link_button("Log in with Cognito", login_url)
            st.info("Please log in using the button in the sidebar to access the RAG Service.")
            return
        else:
            st.sidebar.success("Authenticated")
            if st.sidebar.button("Log out"):
                for key in ("id_token", "access_token", "refresh_token", "messages"):
                    st.session_state.pop(key, None)
                st.rerun()

    page = st.sidebar.radio(
        "Navigate",
        ["💬 Chat", "📤 Upload", "📁 Documents"],
        index=0,
    )

    if page == "💬 Chat":
        _page_chat()
    elif page == "📤 Upload":
        _page_upload()
    elif page == "📁 Documents":
        _page_browser()


if __name__ == "__main__":
    main()
