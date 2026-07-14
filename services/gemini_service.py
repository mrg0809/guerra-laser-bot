"""Gemini via google-genai: extracción técnica (Paso A) y redacción final (Paso C)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Timeout HTTP del SDK en milisegundos (evita congelar el proceso)
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))

EXTRACTION_SYSTEM = """Eres un ingeniero de refacciones para máquinas láser (Guerra Laser, Guadalajara).
Analiza el mensaje del cliente y las imágenes si las hay. Devuelve SOLO un JSON válido con esta forma:
{
  "tipo_producto": "string o null (ej: lente, cadena, acrílico, espejo, tubo)",
  "marca_fuente": "string o null (ej: Raycus, IPG, JPT)",
  "tecnologia_detectada": "CO2 | Fibra | CNC | UV | otro | null (la que el cliente mencionó o se infiere)",
  "potencia_detectada": "string o null (ej: 60W, 80W, 100W)",
  "medidas_o_specs": ["lista de términos como D30, F-theta, paso, diámetro, longitud focal, mm"],
  "terminos_busqueda": ["lista corta de palabras clave para buscar en catálogo, sin relleno"],
  "pide_fotos_o_ver_producto": true o false (true si pide fotos, imágenes, ver producto, mandar foto, etc.),
  "intencion": "INFO | COMPRA | SOPORTE_QUEJA (clasifica por intención principal del cliente)",
  "motivo_handover": "string breve explicando motivo si intencion != INFO, de lo contrario string vacío",
  "notas_tecnicas": "string breve para el equipo interno"
}
Si algo no se deduce, usa null, false o listas vacías. Responde en español en los valores del JSON."""

RESPONSE_SYSTEM = """Eres el asistente técnico-comercial de Guerra Laser (Guadalajara, Jalisco) en WhatsApp.
Identidad transparente obligatoria: eres la Inteligencia Artificial de Guerra Laser, no un humano.

Estilo obligatorio:
- Breve: máximo 2 o 3 párrafos cortos; menos es más.
- Contextual: si el cliente ya indicó potencia o tecnología (ej. 60W, CO2), NO ofrezcas productos de otras categorías (Fibra, CNC, etc.) salvo que sea imprescindible para resolver su duda.
- No repetitivo: no saludes en cada mensaje (ni "Hola/Buen día" repetidos). Si el mensaje parece seguimiento de una charla, continúa directo al tema.
- Visual: si en el contexto hay URLs de imágenes del catálogo y el cliente pidió ver fotos, confirma brevemente que compartes las imágenes (el sistema las puede adjuntar); no pegues listas largas de URLs si ya se enviarán como archivos.
- Enlaces web: si en el contexto vienen `url_detalle_web` en productos o `url_categoria_web` en categorías, incluye al menos un enlace relevante para que el cliente vea fichas o listados en guerralaser.com (formato corto, una URL por producto o categoría destacada). No inventes rutas ni dominios; solo usa las URLs del JSON.
- Vendedor: cierra SIEMPRE con una pregunta concreta que invite a la acción (ej. cotizar, elegir medida, confirmar modelo).

Sé preciso con medidas y compatibilidades; no inventes datos que no estén en el catálogo o en el mensaje del cliente."""


def _fallback_extraction(raw: str = "", motivo: str = "") -> dict[str, Any]:
    return {
        "tipo_producto": None,
        "marca_fuente": None,
        "tecnologia_detectada": None,
        "potencia_detectada": None,
        "medidas_o_specs": [],
        "terminos_busqueda": [],
        "pide_fotos_o_ver_producto": False,
        "intencion": "INFO",
        "motivo_handover": motivo or "",
        "notas_tecnicas": (raw[:2000] if raw else "Error o timeout con Gemini"),
    }


class GeminiService:
    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        if not api_key:
            raise ValueError("GOOGLE_API_KEY es requerida")
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        self._model_name = model_name or MODEL_NAME

    def _fetch_images_sync(self, urls: list[str], timeout: float = 10.0) -> list[types.Part]:
        parts: list[types.Part] = []
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for url in urls:
                try:
                    r = client.get(url)
                    r.raise_for_status()
                    ctype = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                    if not ctype.startswith("image/"):
                        ctype = "image/jpeg"
                    parts.append(types.Part.from_bytes(data=r.content, mime_type=ctype))
                except Exception as e:
                    logger.warning("No se pudo cargar imagen %s: %s", url, e)
        return parts

    def extract_technical_specs(
        self,
        user_text: str,
        image_urls: list[str],
    ) -> dict[str, Any]:
        """Paso A: texto + imágenes -> especificaciones estructuradas."""
        prompt = (
            f"Mensaje del cliente:\n{user_text or '(sin texto, solo imágenes)'}\n\n"
            "Extrae el JSON solicitado."
        )
        image_parts = self._fetch_images_sync(image_urls, timeout=10.0)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=EXTRACTION_SYSTEM,
            temperature=0.1,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        contents: list[Any] = [prompt]
        contents.extend(image_parts)

        raw = ""
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )
            raw = (response.text or "").strip()
            return json.loads(raw)
        except TimeoutError:
            logger.error(
                "TIMEOUT: Gemini tardó demasiado en extract_technical_specs (>%sms)",
                GEMINI_TIMEOUT_MS,
            )
            return _fallback_extraction(motivo="Timeout de Gemini en extracción")
        except httpx.TimeoutException:
            logger.error(
                "TIMEOUT httpx: Gemini tardó demasiado en extract_technical_specs (>%sms)",
                GEMINI_TIMEOUT_MS,
            )
            return _fallback_extraction(motivo="Timeout de Gemini en extracción")
        except json.JSONDecodeError:
            logger.warning("Gemini extracción no devolvió JSON válido: %s", raw[:500])
            return _fallback_extraction(raw=raw)
        except Exception as e:
            logger.error("Error en extract_technical_specs: %s", e)
            return _fallback_extraction(raw=raw, motivo="Error en backend con Gemini")

    def compose_final_reply(
        self,
        user_text: str,
        extracted: dict[str, Any],
        productos_db: list[dict[str, Any]],
        categorias_relacionadas: list[dict[str, Any]],
        product_media: list[dict[str, Any]],
        image_urls: list[str],
        fragmentos_en_este_lote: int,
        should_introduce_ai_identity: bool,
        conversation_memory: dict[str, Any] | None = None,
    ) -> str:
        """Paso C: contexto + resultados de BD -> respuesta al cliente."""
        ctx = {
            "mensaje_cliente": user_text,
            "fragmentos_acumulados_en_este_lote": fragmentos_en_este_lote,
            "especificaciones_extraidas": extracted,
            "productos_encontrados": productos_db,
            "categorias_relacionadas": categorias_relacionadas,
            "medios_de_producto": product_media,
            "memoria_conversacion": conversation_memory or {},
        }
        prompt = (
            "Redacta la respuesta al cliente usando el contexto siguiente. "
            "Si fragmentos_acumulados_en_este_lote es mayor que 1, asume seguimiento inmediato y evita saludo inicial. "
            f"Identidad en esta respuesta: {'PRESENTARTE como la Inteligencia Artificial de Guerra Laser' if should_introduce_ai_identity else 'NO presentarte de nuevo como IA; continúa natural'} . "
            "Si el cliente escribe algo corto de seguimiento (ej: 'tendrás fotos?', 'precio?', 'y compatibilidad?'), usa primero la memoria_conversacion y productos previos antes de pedirle que repita datos. "
            "Si no hay productos pero sí categorías con url_categoria_web, orienta con el enlace de la categoría. "
            "Si el catálogo está vacío, indica que no hubo coincidencias y pide datos o ofrece asesoría sin inventar referencias.\n\n"
            f"CONTEXTO (JSON):\n{json.dumps(ctx, ensure_ascii=False, default=str)}"
        )
        image_parts = self._fetch_images_sync(image_urls, timeout=10.0)
        contents: list[Any] = [prompt]
        contents.extend(image_parts)
        config = types.GenerateContentConfig(
            system_instruction=RESPONSE_SYSTEM,
            temperature=0.4,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )
            text = (response.text or "").strip()
            if text:
                return text
        except TimeoutError:
            logger.error(
                "TIMEOUT: Gemini tardó demasiado en compose_final_reply (>%sms)",
                GEMINI_TIMEOUT_MS,
            )
        except httpx.TimeoutException:
            logger.error(
                "TIMEOUT httpx: Gemini tardó demasiado en compose_final_reply (>%sms)",
                GEMINI_TIMEOUT_MS,
            )
        except Exception as e:
            logger.error("Error al componer respuesta final (Paso C): %s", e)

        return (
            "Gracias por tu paciencia. En este momento presenté un pequeño inconveniente técnico "
            "para procesar la solicitud, pero un asesor humano de Guerra Laser te atenderá de inmediato."
        )
