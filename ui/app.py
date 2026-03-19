import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def render_answer(data: dict):
    """Render the answer with confidence and sources."""
    confidence = data["confidence"]
    score = confidence["score"]
    level = confidence["level"]

    # Answer
    st.subheader("Answer")
    st.markdown(data["answer"])

    # Confidence bar
    st.subheader("Confidence")
    st.progress(score, text=f"{level.upper()} ({score:.1%})")

    # Confidence components
    with st.expander("Confidence Breakdown"):
        components = confidence["components"]
        cols = st.columns(len(components))
        for i, (name, value) in enumerate(components.items()):
            with cols[i]:
                label = name.replace("_", " ").title()
                st.metric(label, f"{value:.2f}")

    # Guardrails
    guardrails = data.get("guardrails", {})
    with st.expander("Guardrail Status"):
        st.write(f"- **Retrieval Quality:** {guardrails.get('retrieval_quality', 'n/a')}")
        st.write(f"- **Grounding Check:** {guardrails.get('grounding_check', 'n/a')}")
        if guardrails.get("reason"):
            st.warning(guardrails["reason"])

    # Sources
    sources = data.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)} chunks)"):
            for i, src in enumerate(sources):
                st.markdown(f"**Source {i+1}** (Page {src['page_number']}, Section: {src['section'] or 'General'})")
                st.code(src["text"], language=None)
                st.divider()


def render_extraction(data: dict):
    """Render structured extraction results."""
    shipment = data["shipment_data"]
    missing = data["missing_fields"]
    conf = data["extraction_confidence"]

    st.subheader("Extraction Results")
    st.progress(conf, text=f"Extraction Confidence: {conf:.0%} ({len(missing)} missing fields)")

    col1, col2 = st.columns(2)

    fields = [
        ("Shipment ID", "shipment_id"),
        ("Shipper", "shipper"),
        ("Consignee", "consignee"),
        ("Pickup Date/Time", "pickup_datetime"),
        ("Delivery Date/Time", "delivery_datetime"),
        ("Equipment Type", "equipment_type"),
        ("Mode", "mode"),
        ("Rate", "rate"),
        ("Currency", "currency"),
        ("Weight", "weight"),
        ("Carrier Name", "carrier_name"),
    ]

    for i, (label, key) in enumerate(fields):
        target = col1 if i % 2 == 0 else col2
        value = shipment.get(key)
        with target:
            if value is not None:
                if key == "rate":
                    st.metric(label, f"${value:,.2f}" if isinstance(value, (int, float)) else str(value))
                else:
                    st.metric(label, str(value))
            else:
                st.metric(label, "Not Found", delta="missing", delta_color="inverse")

    with st.expander("Raw JSON"):
        st.json(shipment)

    if missing:
        st.warning(f"Missing fields: {', '.join(missing)}")


# --- Page Config ---
st.set_page_config(page_title="Ultra Doc-Intelligence", layout="wide")
st.title("Ultra Doc-Intelligence")
st.caption("Logistics Document Q&A with Structured Extraction")

tab1, tab2, tab3 = st.tabs(["Upload Document", "Ask Questions", "Extract Data"])

# --- Tab 1: Upload ---
with tab1:
    st.header("Upload a Logistics Document")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        help="Upload a Rate Confirmation, BOL, Invoice, or Shipment Instructions",
    )

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Parsing, chunking, and embedding..."):
            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["document_id"] = data["document_id"]
                    st.session_state["filename"] = data["filename"]
                    st.success(
                        f"Document processed successfully!\n\n"
                        f"- **Document ID:** `{data['document_id']}`\n"
                        f"- **Chunks created:** {data['num_chunks']}\n"
                        f"- **Pages:** {data['num_pages']}"
                    )
                else:
                    try:
                        detail = resp.json().get("detail", "Unknown error")
                    except Exception:
                        detail = resp.text or f"HTTP {resp.status_code}"
                    st.error(f"Error: {detail}")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                st.error("Could not connect to the API. Make sure the backend is running on " + API_URL)

    if "document_id" in st.session_state:
        st.info(f"Active document: **{st.session_state.get('filename', '')}** (`{st.session_state['document_id'][:8]}...`)")

# --- Tab 2: Ask Questions ---
with tab2:
    st.header("Ask Questions About Your Document")

    if "document_id" not in st.session_state:
        st.warning("Please upload a document first.")
    else:
        st.info(f"Querying: **{st.session_state.get('filename', '')}**")

        with st.form("ask_form", clear_on_submit=False):
            question = st.text_input(
                "Your question",
                placeholder="e.g., What is the carrier rate? Who is the consignee?",
            )
            submitted = st.form_submit_button("Ask", type="primary")

        if submitted and question:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "document_id": st.session_state["document_id"],
                            "question": question,
                        },
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        render_answer(data)
                    elif resp.status_code == 404:
                        st.error("Document not found. Please re-upload.")
                    else:
                        try:
                            detail = resp.json().get("detail", "Unknown error")
                        except Exception:
                            detail = resp.text or f"HTTP {resp.status_code}"
                        st.error(f"Error: {detail}")
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    st.error("Could not connect to the API.")

# --- Tab 3: Extract Data ---
with tab3:
    st.header("Structured Shipment Data Extraction")

    if "document_id" not in st.session_state:
        st.warning("Please upload a document first.")
    else:
        st.info(f"Extracting from: **{st.session_state.get('filename', '')}**")

        if st.button("Extract Shipment Data", type="primary"):
            with st.spinner("Extracting structured data..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/extract",
                        json={"document_id": st.session_state["document_id"]},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        render_extraction(data)
                    elif resp.status_code == 404:
                        st.error("Document not found. Please re-upload.")
                    else:
                        try:
                            detail = resp.json().get("detail", "Unknown error")
                        except Exception:
                            detail = resp.text or f"HTTP {resp.status_code}"
                        st.error(f"Error: {detail}")
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    st.error("Could not connect to the API.")
