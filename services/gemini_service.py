"""Gemini 1.5 Flash: extracción técnica (Paso A) y redacción final (Paso C)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import google.generativeai as genai
import httpx

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

EXTRACTION_SYSTEM = """Eres un ingeniero de refacciones para máquinas láser (Guerra Laser, Guadalajara).
Analiza el mensaje del cliente y las imágenes si las hay. Devuelve SOLO un JSON válido con esta forma:
{
  "tipo_producto": "string o null (ej: lente, cadena, acrílico, espejo, tubo)",
  "marca_fuente": "string o null (ej: Raycus, IPG, JPT)",
  "medidas_o_specs": ["lista de términos como D30, F-theta, paso, diámetro, longitud focal, mm"],
  "terminos_busqueda": ["lista corta de palabras clave para buscar en catálogo, sin relleno"],
  "notas_tecnicas": "string breve para el equipo interno"
}
Si algo no se deduce, usa null o listas vacías. Responde en español en los valores del JSON."""

RESPONSE_SYSTEM = """Eres el asistente técnico-comercial de Guerra Laser (Guadalajara, Jalisco).
Tono: experto, servicial, claro; preciso con medidas y compatibilidades; sin inventar datos que no estén en el catálogo o en el mensaje.
Puedes orientar al cliente y sugerir siguiente paso (cotización, medición, compatibilidad) cuando aplique.
No uses jerga innecesaria; sé amable y profesional."""


class GeminiService:
    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        if not api_key:
            raise ValueError("GOOGLE_API_KEY es requerida")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name or MODEL_NAME)

    def _fetch_images_sync(self, urls: list[str], timeout: float = 20.0) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for url in urls:
                try:
                    r = client.get(url)
                    r.raise_for_status()
                    ctype = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                    if not ctype.startswith("image/"):
                        ctype = "image/jpeg"
                    parts.append({"mime_type": ctype, "data": r.content})
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
        image_parts = self._fetch_images_sync(image_urls)
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        contents: list[Any] = [EXTRACTION_SYSTEM + "\n\n" + prompt]
        contents.extend(image_parts)
        response = self._model.generate_content(
            contents,
            generation_config=generation_config,
        )
        raw = (response.text or "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Gemini extracción no devolvió JSON válido: %s", raw[:500])
            return {
                "tipo_producto": None,
                "marca_fuente": None,
                "medidas_o_specs": [],
                "terminos_busqueda": [],
                "notas_tecnicas": raw[:2000],
            }

    def compose_final_reply(
        self,
        user_text: str,
        extracted: dict[str, Any],
        productos_db: list[dict[str, Any]],
        image_urls: list[str],
    ) -> str:
        """Paso C: contexto + resultados de BD -> respuesta al cliente."""
        ctx = {
            "mensaje_cliente": user_text,
            "especificaciones_extraidas": extracted,
            "productos_encontrados": productos_db,
        }
        prompt = (
            f"{RESPONSE_SYSTEM}\n\n"
            "Redacta la respuesta al cliente usando el contexto siguiente. "
            "Si el catálogo está vacío, indica que no hubo coincidencias y pide datos o ofrece asesoría sin inventar referencias.\n\n"
            f"CONTEXTO (JSON):\n{json.dumps(ctx, ensure_ascii=False, default=str)}"
        )
        image_parts = self._fetch_images_sync(image_urls)
        contents: list[Any] = [prompt]
        contents.extend(image_parts)
        generation_config = genai.GenerationConfig(temperature=0.4)
        response = self._model.generate_content(
            contents,
            generation_config=generation_config,
        )
        return (response.text or "").strip() or (
            "Gracias por escribirnos. En este momento no pude generar una respuesta; "
            "un asesor de Guerra Laser te contactará en breve."
        )
