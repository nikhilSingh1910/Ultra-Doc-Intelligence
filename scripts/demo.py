#!/usr/bin/env python3
"""End-to-end demo and evaluation script for Ultra Doc-Intelligence.

Uploads real logistics PDFs, asks questions with known answers,
runs structured extraction, and reports accuracy. Use this to
validate the full pipeline before submission.

Usage:
    # Start the backend first:
    uvicorn src.main:app --port 8000

    # Then run this script:
    python scripts/demo.py
"""

import json
import sys
import time

import requests

API_URL = "http://localhost:8000"

# ─── Test documents ───────────────────────────────────────────────
DOCUMENTS = [
    {
        "file": "tests/fixtures/LD53657-Carrier-RC.pdf",
        "name": "Carrier Rate Confirmation",
        "questions": [
            {
                "q": "What is the reference ID or load number?",
                "expected_contains": ["LD53657"],
            },
            {
                "q": "Who is the carrier?",
                "expected_contains": ["SWIFT SHIFT LOGISTICS"],
            },
            {
                "q": "What is the carrier rate or agreed amount?",
                "expected_contains": ["400"],
            },
            {
                "q": "What equipment type is being used?",
                "expected_contains": ["Flatbed"],
            },
            {
                "q": "What is the shipping date?",
                "expected_contains": ["02-08-2026"],
            },
            {
                "q": "What is the commodity being shipped?",
                "expected_contains": ["Ceramic"],
            },
            {
                "q": "What is the total weight?",
                "expected_contains": ["56,000"],
            },
            {
                "q": "Who is the driver?",
                "expected_contains": ["John Doe"],
            },
        ],
        "extraction_expected": {
            "shipment_id": "LD53657",
            "carrier_name": "SWIFT SHIFT LOGISTICS",
            "equipment_type": "Flatbed",
            "mode": "FTL",
            "weight": "56000",
        },
    },
    {
        "file": "tests/fixtures/BOL53657_billoflading.pdf",
        "name": "Bill of Lading",
        "questions": [
            {
                "q": "What is the load ID?",
                "expected_contains": ["LD53657"],
            },
            {
                "q": "Who is the shipper?",
                "expected_contains": ["AAA"],
            },
            {
                "q": "Who is the consignee?",
                "expected_contains": ["xyz"],
            },
            {
                "q": "What is the weight?",
                "expected_contains": ["56000"],
            },
            {
                "q": "What is the ship date?",
                "expected_contains": ["02-08-2026"],
            },
        ],
        "extraction_expected": {
            "shipment_id": "LD53657",
            "shipper": "AAA",
            "consignee": "xyz",
            "weight": "56000",
        },
    },
    {
        "file": "tests/fixtures/LD53657-Shipper-RC.pdf",
        "name": "Shipper Rate Confirmation",
        "questions": [
            {
                "q": "What is the agreed amount or customer rate?",
                "expected_contains": ["1000"],
            },
            {
                "q": "What is the reference ID?",
                "expected_contains": ["LD53657"],
            },
            {
                "q": "What is the load type?",
                "expected_contains": ["FTL"],
            },
        ],
        "extraction_expected": {
            "shipment_id": "LD53657",
            "rate": "1000",
            "mode": "FTL",
        },
    },
]

# ─── Guardrail test: out-of-scope question ────────────────────────
OUT_OF_SCOPE_QUESTIONS = [
    "What do you think about artificial intelligence?",
    "Write me a poem about trucks",
]


def main():
    print("=" * 70)
    print("ULTRA DOC-INTELLIGENCE — END-TO-END EVALUATION")
    print("=" * 70)

    # Check API is running
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"\n✓ API is healthy at {API_URL}")
    except Exception as e:
        print(f"\n✗ API not reachable at {API_URL}: {e}")
        print("  Start the backend first: uvicorn src.main:app --port 8000")
        sys.exit(1)

    total_questions = 0
    correct_answers = 0
    total_extraction_fields = 0
    correct_extraction_fields = 0
    guardrail_passes = 0
    guardrail_tests = 0

    for doc in DOCUMENTS:
        print(f"\n{'─' * 70}")
        print(f"DOCUMENT: {doc['name']}")
        print(f"{'─' * 70}")

        # ── Upload ──
        with open(doc["file"], "rb") as f:
            resp = requests.post(
                f"{API_URL}/upload",
                files={"file": (doc["file"].split("/")[-1], f, "application/pdf")},
                timeout=120,
            )

        if resp.status_code != 200:
            print(f"  ✗ Upload failed: {resp.status_code} {resp.text}")
            continue

        upload_data = resp.json()
        doc_id = upload_data["document_id"]
        print(f"  ✓ Uploaded: {upload_data['num_chunks']} chunks, {upload_data['num_pages']} pages")

        # ── Ask Questions ──
        print(f"\n  Q&A ({len(doc['questions'])} questions):")
        for qa in doc["questions"]:
            total_questions += 1
            resp = requests.post(
                f"{API_URL}/ask",
                json={"document_id": doc_id, "question": qa["q"]},
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"    ✗ [{qa['q']}] API error: {resp.status_code}")
                continue

            data = resp.json()
            answer = data["answer"]
            conf = data["confidence"]
            sources = data["sources"]

            # Check if expected content is in the answer
            answer_lower = answer.lower()
            found = all(exp.lower() in answer_lower for exp in qa["expected_contains"])
            if found:
                correct_answers += 1

            status = "✓" if found else "✗"
            print(f"    {status} Q: {qa['q']}")
            print(f"      A: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print(f"      Confidence: {conf['score']:.2f} ({conf['level']}) | Sources: {len(sources)}")
            if sources:
                print(f"      Source pages: {[s['page_number'] for s in sources]}")
            if not found:
                print(f"      Expected: {qa['expected_contains']}")

        # ── Structured Extraction ──
        print(f"\n  Extraction:")
        resp = requests.post(
            f"{API_URL}/extract",
            json={"document_id": doc_id},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"    ✗ Extraction failed: {resp.status_code}")
            continue

        ext = resp.json()
        shipment = ext["shipment_data"]
        missing = ext["missing_fields"]
        ext_conf = ext["extraction_confidence"]

        print(f"    Confidence: {ext_conf:.2f} | Missing: {missing}")

        for field, expected_val in doc["extraction_expected"].items():
            total_extraction_fields += 1
            actual = str(shipment.get(field) or "")
            match = expected_val.lower() in actual.lower()
            if match:
                correct_extraction_fields += 1
            status = "✓" if match else "✗"
            print(f"    {status} {field}: {actual or 'null'} (expected: {expected_val})")

    # ── Guardrail Tests ──
    if total_questions > 0:  # Only test guardrails if we have a document
        print(f"\n{'─' * 70}")
        print("GUARDRAIL TESTS")
        print(f"{'─' * 70}")

        for q in OUT_OF_SCOPE_QUESTIONS:
            guardrail_tests += 1
            resp = requests.post(
                f"{API_URL}/ask",
                json={"document_id": doc_id, "question": q},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                blocked = "scope" in data["answer"].lower() or data["confidence"]["score"] == 0.0
                if blocked:
                    guardrail_passes += 1
                    print(f"  ✓ Blocked: {q[:60]}")
                else:
                    print(f"  ✗ Not blocked: {q[:60]}")
                    print(f"    Got: {data['answer'][:80]}")
            else:
                print(f"  ? API error for guardrail test: {resp.status_code}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")

    qa_accuracy = correct_answers / max(total_questions, 1) * 100
    ext_accuracy = correct_extraction_fields / max(total_extraction_fields, 1) * 100
    guard_rate = guardrail_passes / max(guardrail_tests, 1) * 100

    print(f"  Q&A Accuracy:        {correct_answers}/{total_questions} ({qa_accuracy:.0f}%)")
    print(f"  Extraction Accuracy: {correct_extraction_fields}/{total_extraction_fields} ({ext_accuracy:.0f}%)")
    print(f"  Guardrail Pass Rate: {guardrail_passes}/{guardrail_tests} ({guard_rate:.0f}%)")
    print()

    if qa_accuracy >= 80 and ext_accuracy >= 80 and guard_rate >= 80:
        print("  VERDICT: ✓ PASS — System meets quality thresholds")
    elif qa_accuracy >= 60 and ext_accuracy >= 60:
        print("  VERDICT: ~ PARTIAL — Some accuracy gaps to investigate")
    else:
        print("  VERDICT: ✗ FAIL — Significant accuracy issues")

    print()


if __name__ == "__main__":
    main()
