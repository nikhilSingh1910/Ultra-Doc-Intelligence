"""OpenAI LLM adapter for document Q&A and structured extraction.

Handles two distinct use-cases:
1. **ask**: RAG-grounded Q&A — answers strictly from retrieved context.
2. **extract**: Structured field extraction via function calling (tool_use).

Prompts are crafted with logistics domain knowledge to handle
Rate Confirmations, BOLs, Freight Invoices, and Shipment Instructions.
"""

import json
from typing import Any, Dict

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import get_settings
from src.exceptions import LLMServiceError
from src.util.logging_setup import get_logger

logger = get_logger(__name__)

_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)

# ---------------------------------------------------------------------------
# System prompts — logistics domain-aware
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ASK = """You are an expert logistics document analyst embedded in a Transportation Management System (TMS). You answer questions about shipping documents — Rate Confirmations, Bills of Lading (BOL), Freight Invoices, Shipment Instructions, and Proof of Delivery (POD).

STRICT RULES:
1. Answer ONLY from the document context below. Never use outside knowledge.
2. If the answer is not in the context, respond exactly: "Not found in document."
3. Quote values exactly as they appear — dollar amounts, dates, reference numbers, weights.
4. When a question is ambiguous (e.g., "what is the rate?" could mean line haul, fuel surcharge, or total), answer with the most prominent/primary value and note any related figures.

LOGISTICS FIELD KNOWLEDGE:
- Shipper = the party shipping goods (SHIP FROM). Not the broker or carrier.
- Consignee = the receiving party (SHIP TO / DELIVER TO).
- Carrier = the trucking company performing the haul.
- BOL# / PRO# / PO# = document reference numbers, each serves a different purpose.
- MC# = Motor Carrier authority number. DOT# = Department of Transportation number.
- Rate typically refers to the line haul rate unless "total" is specified.
- Equipment types: 53' Dry Van, Reefer (temperature controlled), Flatbed, Step Deck, etc.
- Modes: FTL (Full Truckload), LTL (Less Than Truckload), Intermodal, Partial.

Document Context:
{context}
"""

SYSTEM_PROMPT_EXTRACT = """You are a logistics data extraction engine. Extract structured shipment data from the document below.

EXTRACTION RULES:
1. Return values EXACTLY as they appear in the document. Do not reformat or infer.
2. For missing fields, return null — never fabricate values.
3. "shipper" = the SHIP FROM / SHIPPER party, not the broker or freight payer.
4. "consignee" = the SHIP TO / CONSIGNEE / DELIVER TO party.
5. "rate" = the LINE HAUL or base carrier rate as a number (no $ sign). If only a total is shown, use that.
6. "weight" = include the unit exactly as written (e.g., "42,000 lbs").
7. "pickup_datetime" and "delivery_datetime" = include date and time if both are present.
8. "equipment_type" = the trailer/equipment specification (e.g., "53' Dry Van", "48' Reefer").
9. "mode" = shipping mode: FTL, LTL, Intermodal, Partial, etc.
10. "shipment_id" = the primary reference: Load Number, Shipment ID, or Reference Number (not BOL# or PRO#).
11. "carrier_name" = the trucking company, not the broker or third-party logistics provider.
12. "currency" = ISO 4217 code (USD, CAD, MXN). Default to USD if amounts use $ with no explicit currency.
"""

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_shipment_data",
        "description": "Extract structured shipment data from a logistics document.",
        "parameters": {
            "type": "object",
            "properties": {
                "shipment_id": {
                    "type": ["string", "null"],
                    "description": "Primary reference: Load Number, Shipment ID, or Reference Number",
                },
                "shipper": {
                    "type": ["string", "null"],
                    "description": "SHIP FROM party name (not broker)",
                },
                "consignee": {
                    "type": ["string", "null"],
                    "description": "SHIP TO / DELIVER TO party name",
                },
                "pickup_datetime": {
                    "type": ["string", "null"],
                    "description": "Pickup date and time as written in the document",
                },
                "delivery_datetime": {
                    "type": ["string", "null"],
                    "description": "Delivery date and time as written in the document",
                },
                "equipment_type": {
                    "type": ["string", "null"],
                    "description": "Trailer/equipment spec (e.g., 53' Dry Van, Reefer, Flatbed)",
                },
                "mode": {
                    "type": ["string", "null"],
                    "description": "Shipping mode: FTL, LTL, Intermodal, Partial",
                },
                "rate": {
                    "type": ["number", "null"],
                    "description": "Line haul / base carrier rate as a number",
                },
                "currency": {
                    "type": ["string", "null"],
                    "description": "ISO 4217 currency code (e.g., USD, CAD)",
                },
                "weight": {
                    "type": ["string", "null"],
                    "description": "Total weight with unit (e.g., '42,000 lbs')",
                },
                "carrier_name": {
                    "type": ["string", "null"],
                    "description": "Trucking company name (not broker or 3PL)",
                },
            },
            "required": [
                "shipment_id", "shipper", "consignee",
                "pickup_datetime", "delivery_datetime",
                "equipment_type", "mode", "rate", "currency",
                "weight", "carrier_name",
            ],
        },
    },
}


class LLMClient:
    """OpenAI chat completion adapter for logistics document intelligence.

    Uses the async client so LLM calls don't block the event loop.
    GPT-4o-mini for fast Q&A and GPT-4o for higher-accuracy extraction.
    All calls include exponential backoff retry on transient errors.
    """

    def __init__(
        self,
        model: str | None = None,
        extraction_model: str | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        self._model = model or settings.llm_model
        self._extraction_model = extraction_model or settings.extraction_model
        self._client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def ask(self, question: str, context: str) -> str:
        """Answer a question grounded in the provided document context.

        The system prompt enforces strict grounding — the LLM will refuse
        to answer if the information is not present in context.
        """
        system_message = SYSTEM_PROMPT_ASK.format(context=context)
        logger.info("LLM ask: question=%r, context_len=%d", question[:80], len(context))

        content = await self._chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            max_tokens=1024,
        )
        return content

    async def extract(self, document_text: str) -> Dict[str, Any]:
        """Extract structured shipment data using function calling.

        Forces the ``extract_shipment_data`` tool so the response is
        guaranteed to be valid JSON matching the schema.
        """
        logger.info("LLM extract: doc_len=%d", len(document_text))

        response = await self._chat_raw(
            model=self._extraction_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXTRACT},
                {"role": "user", "content": f"Extract shipment data from this document:\n\n{document_text}"},
            ],
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_shipment_data"}},
            max_tokens=2048,
        )

        choice = response.choices[0]
        if choice.message.tool_calls:
            args = choice.message.tool_calls[0].function.arguments
            return json.loads(args)

        # Fallback: try to parse content as JSON
        content = choice.message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM extraction returned non-JSON content, returning empty dict")
            return {}

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _chat(self, model: str, messages: list, max_tokens: int = 1024) -> str:
        """Call chat completion with retry, return text content."""
        try:
            response = await self._client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except _RETRYABLE:
            raise
        except openai.OpenAIError as exc:
            raise LLMServiceError(str(exc)) from exc

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _chat_raw(self, model: str, messages: list, **kwargs):
        """Call chat completion with retry, return raw response."""
        try:
            return await self._client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, **kwargs,
            )
        except _RETRYABLE:
            raise
        except openai.OpenAIError as exc:
            raise LLMServiceError(str(exc)) from exc
