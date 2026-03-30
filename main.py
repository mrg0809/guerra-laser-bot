"""
Bot Chatwoot + FastAPI: debounce, Gemini (extracción + respuesta), Supabase productos.
"""

from __future__ import annotations

import logging
import os
import re
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
PRODUCT_MEDIA_TABLE = os.getenv("PRODUCT_MEDIA_TABLE", "product_media")
CATEGORIES_TABLE = os.getenv("CATEGORIES_TABLE", "categories")
# Columnas para búsqueda tipo ILIKE (ajusta a tu esquema)
COL_NOMBRE = os.getenv("PRODUCTOS_COL_NOMBRE", "nombre")
COL_DESC = os.getenv("PRODUCTOS_COL_DESCRIPCION", "descripcion")
COL_SKU = os.getenv("PRODUCTOS_COL_SKU", "sku")
COL_SLUG = os.getenv("PRODUCTOS_COL_SLUG", "slug")
COL_CATEGORY_ID = os.getenv("PRODUCTOS_COL_CATEGORY_ID", "category_id")

COL_CAT_NOMBRE = os.getenv("CATEGORIES_COL_NOMBRE", "name")
COL_CAT_SLUG = os.getenv("CATEGORIES_COL_SLUG", "slug")

# Sitio público (enlaces para el cliente)
PUBLIC_SITE_BASE = os.getenv("PUBLIC_SITE_BASE", "https://guerralaser.com").rstrip("/")
PUBLIC_PRODUCT_PATH_TEMPLATE = os.getenv("PUBLIC_PRODUCT_PATH_TEMPLATE", "/productos/{slug}")
PUBLIC_CATEGORY_PATH_TEMPLATE = os.getenv("PUBLIC_CATEGORY_PATH_TEMPLATE", "/categorias/{slug}")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

_supabase: Client | None = None
_gemini: GeminiService | None = None
debouncer = ConversationDebouncer(delay_seconds=DEBOUNCE_SECONDS)
_handover_conversations: set[int] = set()


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


def build_public_url(path_template: str, slug: str | None) -> str | None:
    """Arma https://guerralaser.com + ruta con slug (plantillas PUBLIC_*_PATH_TEMPLATE)."""
    if not slug or not isinstance(slug, str):
        return None
    s = slug.strip().strip("/")
    if not s:
        return None
    try:
        path = path_template.format(slug=s)
    except KeyError:
        path = path_template.replace("{slug}", s)
    if not path.startswith("/"):
        path = "/" + path
    return f"{PUBLIC_SITE_BASE}{path}"


def enrich_product_with_public_url(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    slug = out.get(COL_SLUG) or out.get("slug")
    out["url_detalle_web"] = build_public_url(PUBLIC_PRODUCT_PATH_TEMPLATE, slug if isinstance(slug, str) else None)
    return out


def enrich_category_with_public_url(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    slug = out.get(COL_CAT_SLUG) or out.get("slug")
    out["url_categoria_web"] = build_public_url(PUBLIC_CATEGORY_PATH_TEMPLATE, slug if isinstance(slug, str) else None)
    return out


def _merge_categorias_por_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        rid = r.get("id")
        key = str(rid) if rid is not None else ""
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(r)
    return out


def fetch_categories_with_urls_by_product_ids(productos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids: list[str] = []
    for p in productos:
        cid = p.get(COL_CATEGORY_ID) or p.get("category_id")
        if cid is not None:
            ids.append(str(cid))
    if not ids:
        return []
    sb = get_supabase()
    try:
        res = sb.table(CATEGORIES_TABLE).select("*").in_("id", list(dict.fromkeys(ids))).execute()
    except Exception as e:
        logger.warning("Error Supabase categories by id: %s", e)
        return []
    return [enrich_category_with_public_url(dict(r)) for r in (res.data or [])]


def _or_ilike_categoria(term: str) -> str:
    t = _escape_ilike(term.strip())
    if len(t) < 2:
        return ""
    pat = f"%{t}%"
    return f"{COL_CAT_SLUG}.ilike.{pat},{COL_CAT_NOMBRE}.ilike.{pat}"


def search_categorias(terms: list[str], limit_per_term: int = 8, max_total: int = 10) -> list[dict[str, Any]]:
    """Búsqueda por slug/nombre cuando no hay productos o para reforzar contexto."""
    if not terms:
        return []
    sb = get_supabase()
    seen: set[Any] = set()
    out: list[dict[str, Any]] = []
    for term in terms:
        if len(out) >= max_total:
            break
        filt = _or_ilike_categoria(term)
        if not filt:
            continue
        try:
            res = (
                sb.table(CATEGORIES_TABLE)
                .select("*")
                .or_(filt)
                .limit(limit_per_term)
                .execute()
            )
            rows = res.data or []
        except Exception as e:
            logger.warning("Error Supabase categories ilike (%s): %s", term, e)
            continue
        for row in rows:
            rid = row.get("id") or id(row)
            if rid in seen:
                continue
            seen.add(rid)
            out.append(enrich_category_with_public_url(dict(row)))
            if len(out) >= max_total:
                break
    return out


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


def fetch_product_media_for_products(productos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filas de `product_media` para los product_id encontrados."""
    ids: list[str] = []
    for p in productos:
        pid = p.get("id")
        if pid is not None:
            ids.append(str(pid))
    if not ids:
        return []
    sb = get_supabase()
    try:
        res = sb.table(PRODUCT_MEDIA_TABLE).select("*").in_("product_id", ids).execute()
    except Exception as e:
        logger.warning("Error Supabase product_media: %s", e)
        return []
    rows: list[dict[str, Any]] = list(res.data or [])

    def _sort_key(r: dict[str, Any]) -> tuple[int, int]:
        primary = 0 if r.get("is_primary") else 1
        order = r.get("display_order")
        try:
            o = int(order) if order is not None else 999
        except (TypeError, ValueError):
            o = 999
        return (primary, o)

    rows.sort(key=_sort_key)
    return rows


_RE_PIDE_FOTOS = re.compile(
    r"(foto|fotos|imagen|imágenes|fotografía|mandame|mándame|muestra|enseñ|verlo|ver la|ver el|verlas|verlos)",
    re.IGNORECASE,
)


def _quiere_fotos_desde_texto_o_extraccion(user_text: str, extracted: dict[str, Any]) -> bool:
    if extracted.get("pide_fotos_o_ver_producto") is True:
        return True
    t = user_text or ""
    return bool(_RE_PIDE_FOTOS.search(t))


def _urls_media_para_enviar(media_rows: list[dict[str, Any]], max_n: int = 5) -> list[str]:
    out: list[str] = []
    for r in media_rows:
        u = r.get("url") or r.get("thumbnail_url")
        if u and isinstance(u, str) and u not in out:
            out.append(u.strip())
        if len(out) >= max_n:
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
    for k in ("tipo_producto", "marca_fuente", "tecnologia_detectada", "potencia_detectada"):
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


def _intent_from_extraction(extracted: dict[str, Any], user_text: str) -> tuple[str, str]:
    raw_intent = str(extracted.get("intencion") or "INFO").strip().upper()
    intent = "INFO"
    if raw_intent in {"COMPRA", "SOPORTE_QUEJA"}:
        intent = raw_intent

    text = (user_text or "").lower()
    text_norm = (
        text.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
    compra_strong_signals = (
        "quiero comprar",
        "lo quiero",
        "pasame el numero de cuenta",
        "pasame numero de cuenta",
        "pásame el número de cuenta",
        "numero de cuenta",
        "número de cuenta",
        "donde deposito",
        "dónde deposito",
        "te deposito",
        "ya te deposite",
        "ya te deposité",
        "ya pague",
        "ya pagué",
        "como pago",
        "cómo pago",
    )
    soporte_signals = (
        "no funciona",
        "garantia",
        "garantía",
        "molesto",
        "falla",
        "fallando",
        "defecto",
        "reclamo",
        "queja",
    )
    inquiry_signals = (
        "tienes ",
        "tendra",
        "tendrá",
        "manejan ",
        "hay ",
        "precio",
        "cuanto",
        "cuánto",
        "compatib",
        "?",
        "disponible",
    )

    has_soporte_signal = any(x in text_norm for x in soporte_signals)
    has_compra_signal = any(x in text_norm for x in compra_strong_signals)

    # Prioridad de seguridad: señales de soporte/queja siempre ganan.
    if has_soporte_signal:
        intent = "SOPORTE_QUEJA"
    elif intent == "INFO":
        if has_compra_signal:
            intent = "COMPRA"
    else:
        # Guardia: evita falso positivo de COMPRA cuando el cliente solo pregunta disponibilidad/precio/compatibilidad.
        if intent == "COMPRA":
            looks_like_inquiry = any(x in text_norm for x in inquiry_signals)
            if not has_compra_signal and looks_like_inquiry:
                intent = "INFO"

    motivo = str(extracted.get("motivo_handover") or "").strip()
    if not motivo and intent == "COMPRA":
        motivo = "necesitas realizar una compra"
    if not motivo and intent == "SOPORTE_QUEJA":
        motivo = "tienes un problema técnico"
    return intent, motivo


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
    attachment_image_urls: list[str] | None = None,
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
    urls = [u for u in (attachment_image_urls or []) if u][:5]

    if not urls:
        payload = {"content": content, "private": False, "message_type": "outgoing"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code >= 400:
                logger.error("Chatwoot error %s: %s", r.status_code, r.text[:500])
        return

    async with httpx.AsyncClient(timeout=90.0) as client:
        file_fields: list[tuple[str, tuple[str, bytes, str]]] = []
        for idx, img_url in enumerate(urls):
            try:
                ir = await client.get(img_url, follow_redirects=True)
                ir.raise_for_status()
                ctype = ir.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                if not ctype.startswith("image/"):
                    ctype = "image/jpeg"
                ext = ".jpg"
                if "png" in ctype:
                    ext = ".png"
                elif "webp" in ctype:
                    ext = ".webp"
                elif "gif" in ctype:
                    ext = ".gif"
                file_fields.append(
                    ("attachments[]", (f"media_{idx}{ext}", ir.content, ctype)),
                )
            except Exception as e:
                logger.warning("No se pudo descargar imagen para adjuntar en Chatwoot: %s — %s", img_url, e)

        if not file_fields:
            extra = "\n\n" + "\n".join(urls) if content else "\n".join(urls)
            payload = {
                "content": (content or "Te comparto las referencias:") + extra,
                "private": False,
                "message_type": "outgoing",
            }
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code >= 400:
                logger.error("Chatwoot error %s: %s", r.status_code, r.text[:500])
            return

        data = {
            "content": content or "Te comparto las fotos del producto.",
            "private": "false",
            "message_type": "outgoing",
        }
        r = await client.post(url, data=data, files=file_fields, headers=headers)
        if r.status_code >= 400:
            logger.warning(
                "Chatwoot multipart falló (%s), reintento solo texto+URLs: %s",
                r.status_code,
                r.text[:300],
            )
            extra = "\n\n" + "\n".join(urls)
            payload = {
                "content": (content or "") + extra,
                "private": False,
                "message_type": "outgoing",
            }
            r2 = await client.post(url, json=payload, headers=headers)
            if r2.status_code >= 400:
                logger.error("Chatwoot error %s: %s", r2.status_code, r2.text[:500])


async def set_chatwoot_conversation_pending(conversation_id: int, account_id: int | None) -> bool:
    if not CHATWOOT_BASE_URL or not CHATWOOT_API_TOKEN:
        logger.warning("Chatwoot no configurado: no se puede traspasar a humano")
        return False
    aid = account_id
    if aid is None and CHATWOOT_ACCOUNT_ID:
        try:
            aid = int(CHATWOOT_ACCOUNT_ID)
        except ValueError:
            aid = None
    if aid is None:
        logger.warning("account_id ausente; no se pudo cambiar status a pending")
        return False

    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{aid}/conversations/{conversation_id}"
    headers = {"api_access_token": CHATWOOT_API_TOKEN}
    payload = {"status": "pending"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.patch(url, json=payload, headers=headers)
        if r.status_code >= 400:
            logger.error("Error al poner conversación %s en pending (%s): %s", conversation_id, r.status_code, r.text[:500])
            return False
    return True


async def process_conversation(
    conversation_id: int,
    account_id: int | None,
    accumulated: AccumulatedMessage,
) -> None:
    text = accumulated.merge_text()
    images = list(accumulated.image_urls)
    fragmentos = len(accumulated.texts) if accumulated.texts else 1
    gemini = get_gemini()
    extracted = gemini.extract_technical_specs(text, images)
    intent, motivo = _intent_from_extraction(extracted, text)
    if intent in {"COMPRA", "SOPORTE_QUEJA"}:
        if intent == "COMPRA":
            response = (
                "Entendido. He detectado que necesitas realizar una compra. "
                "Para darte una atención segura y personalizada, he notificado al equipo humano de GUERRA LASER "
                "para que retomen esta conversación de inmediato."
            )
        else:
            response = (
                "Entendido. He detectado que tienes un problema técnico. "
                "Para darte una atención segura y personalizada, he notificado al equipo humano de GUERRA LASER "
                "para que retomen esta conversación de inmediato."
            )
        ok = await set_chatwoot_conversation_pending(conversation_id, account_id)
        if ok:
            _handover_conversations.add(conversation_id)
        logger.info("Traspasando conversación %s a humano por motivo: %s", conversation_id, motivo or intent)
        await send_chatwoot_message(conversation_id, account_id, response)
        return

    terms = _terms_from_extraction(extracted)
    productos_raw = search_productos(terms)
    productos = [enrich_product_with_public_url(p) for p in productos_raw]
    categorias_de_productos = fetch_categories_with_urls_by_product_ids(productos_raw)
    categorias_por_terminos = search_categorias(terms) if not productos_raw else []
    categorias_relacionadas = _merge_categorias_por_id(categorias_de_productos + categorias_por_terminos)

    media_rows = fetch_product_media_for_products(productos_raw)
    reply = gemini.compose_final_reply(
        text,
        extracted,
        productos,
        categorias_relacionadas,
        media_rows,
        images,
        fragmentos,
    )
    adjuntos: list[str] = []
    if media_rows and _quiere_fotos_desde_texto_o_extraccion(text, extracted):
        adjuntos = _urls_media_para_enviar(media_rows)
    await send_chatwoot_message(conversation_id, account_id, reply, adjuntos or None)


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
    conversation_obj = body.get("conversation") or (body.get("message") or {}).get("conversation") or {}
    conv_status = (conversation_obj.get("status") or "").lower()
    if conv_status == "pending":
        _handover_conversations.add(conv_id)
        return JSONResponse({"ok": True, "ignored": True, "reason": "pending_human"})
    if conv_id in _handover_conversations:
        return JSONResponse({"ok": True, "ignored": True, "reason": "handover_active"})
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
