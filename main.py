"""
Bot Chatwoot + FastAPI: debounce, Gemini (extracción + respuesta), Supabase productos.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from supabase import Client, create_client

from services.gemini_service import GeminiService
from utils.debouncer import AccumulatedMessage, ConversationDebouncer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = float(os.getenv("DEBOUNCE_SECONDS", "8"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "").rstrip("/")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID", "")

PRODUCTOS_TABLE = os.getenv("PRODUCTOS_TABLE", "productos")
# Columnas para búsqueda tipo ILIKE (ajusta a tu esquema)
COL_NOMBRE = os.getenv("PRODUCTOS_COL_NOMBRE", "nombre")
COL_DESC = os.getenv("PRODUCTOS_COL_DESCRIPCION", "descripcion")
COL_SKU = os.getenv("PRODUCTOS_COL_SKU", "sku")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

_supabase: Client | None = None
_gemini: GeminiService | None = None
debouncer = ConversationDebouncer(delay_seconds=DEBOUNCE_SECONDS)


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL y SUPABASE_KEY son requeridos")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def get_gemini() -> GeminiService:
    global _gemini
    if _gemini is None:
        _gemini = GeminiService(GOOGLE_API_KEY)
    return _gemini


def _escape_ilike(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _or_ilike(term: str) -> str:
    t = _escape_ilike(term.strip())
    if len(t) < 2:
        return ""
    pat = f"%{t}%"
    parts = [
        f"{COL_NOMBRE}.ilike.{pat}",
        f"{COL_DESC}.ilike.{pat}",
    ]
    if COL_SKU:
        parts.append(f"{COL_SKU}.ilike.{pat}")
    return ",".join(parts)


def search_productos(terms: list[str], limit_per_term: int = 12, max_total: int = 25) -> list[dict[str, Any]]:
    """Paso B: búsqueda por texto en `productos` (ILIKE). Ajusta columnas por env."""
    if not terms:
        return []
    sb = get_supabase()
    seen: set[Any] = set()
    out: list[dict[str, Any]] = []
    for term in terms:
        if len(out) >= max_total:
            break
        filt = _or_ilike(term)
        if not filt:
            continue
        try:
            res = (
                sb.table(PRODUCTOS_TABLE)
                .select("*")
                .or_(filt)
                .limit(limit_per_term)
                .execute()
            )
            rows = res.data or []
        except Exception as e:
            logger.warning("Error Supabase OR ilike (%s): %s", term, e)
            continue
        for row in rows:
            rid = row.get("id") or id(row)
            if rid in seen:
                continue
            seen.add(rid)
            out.append(row)
            if len(out) >= max_total:
                break
    return out


def _terms_from_extraction(extracted: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for k in ("terminos_busqueda", "medidas_o_specs"):
        v = extracted.get(k)
        if isinstance(v, list):
            terms.extend(str(x) for x in v if x)
        elif isinstance(v, str) and v.strip():
            terms.append(v.strip())
    for k in ("tipo_producto", "marca_fuente"):
        v = extracted.get(k)
        if v and isinstance(v, str) and v.strip():
            terms.append(v.strip())
    # únicos preservando orden
    seen: set[str] = set()
    uniq: list[str] = []
    for t in terms:
        tl = t.lower().strip()
        if tl and tl not in seen:
            seen.add(tl)
            uniq.append(t.strip())
    return uniq[:12]


def _is_incoming_message(msg: dict[str, Any], body: dict[str, Any]) -> bool:
    mt = msg.get("message_type")
    if body.get("message_type") == "outgoing" or mt == "outgoing" or mt == 1:
        return False
    if body.get("message_type") == "incoming" or mt == "incoming" or mt == 0:
        return True
    sender = msg.get("sender") or {}
    if sender.get("type") == "contact":
        return True
    return False


def parse_chatwoot_incoming(body: dict[str, Any]) -> tuple[str | None, int | None, int | None, list[str]]:
    """
    Devuelve (texto, conversation_id, account_id, urls_imagen).
    Solo mensajes entrantes del contacto.
    """
    event = (body.get("event") or "").lower()
    if event and event not in ("message_created", "message_updated"):
        return None, None, None, []

    msg = body.get("message") or body
    conversation = body.get("conversation") or msg.get("conversation") or {}
    account = body.get("account") or conversation.get("meta") or {}

    if not _is_incoming_message(msg, body):
        return None, None, None, []

    content = msg.get("content") or body.get("content") or ""
    if isinstance(content, dict):
        content = content.get("text") or content.get("body") or ""

    conv_id = conversation.get("id") or body.get("conversation_id") or msg.get("conversation_id")
    acc_id = (
        body.get("account_id")
        or account.get("id")
        or (body.get("account") or {}).get("id")
    )

    try:
        conv_int = int(conv_id) if conv_id is not None else None
    except (TypeError, ValueError):
        conv_int = None
    try:
        acc_int = int(acc_id) if acc_id is not None else None
    except (TypeError, ValueError):
        acc_int = None

    urls: list[str] = []
    for att in msg.get("attachments") or body.get("attachments") or []:
        if not isinstance(att, dict):
            continue
        if (att.get("file_type") or "").lower() != "image":
            continue
        url = (
            att.get("data_url")
            or att.get("download_url")
            or att.get("thumb_url")
            or att.get("url")
        )
        if url:
            urls.append(url)

    text = (content or "").strip() if isinstance(content, str) else ""
    return text or None, conv_int, acc_int, urls


async def send_chatwoot_message(
    conversation_id: int,
    account_id: int | None,
    content: str,
) -> None:
    if not CHATWOOT_BASE_URL or not CHATWOOT_API_TOKEN:
        logger.warning("Chatwoot no configurado: no se envía respuesta")
        return
    aid = account_id
    if aid is None and CHATWOOT_ACCOUNT_ID:
        try:
            aid = int(CHATWOOT_ACCOUNT_ID)
        except ValueError:
            aid = None
    if aid is None:
        logger.warning("account_id ausente; define CHATWOOT_ACCOUNT_ID o inclúyelo en el webhook")
        return
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{aid}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": CHATWOOT_API_TOKEN}
    payload = {"content": content, "private": False, "message_type": "outgoing"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            logger.error("Chatwoot error %s: %s", r.status_code, r.text[:500])


async def process_conversation(
    conversation_id: int,
    account_id: int | None,
    accumulated: AccumulatedMessage,
) -> None:
    text = accumulated.merge_text()
    images = list(accumulated.image_urls)
    gemini = get_gemini()
    extracted = gemini.extract_technical_specs(text, images)
    terms = _terms_from_extraction(extracted)
    productos = search_productos(terms)
    reply = gemini.compose_final_reply(text, extracted, productos, images)
    await send_chatwoot_message(conversation_id, account_id, reply)


async def on_debounced_flush(conversation_key: Any, buf: AccumulatedMessage) -> None:
    if not isinstance(conversation_key, tuple):
        return
    conv_id, acc_id = conversation_key
    if not isinstance(conv_id, int):
        return
    try:
        await process_conversation(conv_id, acc_id, buf)
    except Exception:
        logger.exception("Error procesando conversación %s", conv_id)


app = FastAPI(title="Guerra Laser Chatwoot Bot", version="1.0.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)

    text, conv_id, acc_id, image_urls = parse_chatwoot_incoming(body)
    if conv_id is None:
        return JSONResponse({"ok": True, "ignored": True, "reason": "no conversation"})
    if not text and not image_urls:
        return JSONResponse({"ok": True, "ignored": True, "reason": "empty message"})

    key = (conv_id, acc_id)

    async def flush(_cid: Any, buf: AccumulatedMessage) -> None:
        await on_debounced_flush(_cid, buf)

    await debouncer.push(key, text or "", image_urls, flush)
    return JSONResponse({"ok": True, "debounced": True, "conversation_id": conv_id})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
