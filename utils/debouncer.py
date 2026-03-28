"""Debouncer por conversación: espera un intervalo tras el último mensaje antes de procesar."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Hashable


@dataclass
class AccumulatedMessage:
    """Texto e imágenes acumulados para una conversación."""

    texts: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)

    def merge_text(self) -> str:
        parts = [t.strip() for t in self.texts if t and t.strip()]
        return "\n\n".join(parts) if parts else ""


OnFlush = Callable[[Hashable, AccumulatedMessage], Awaitable[None]]


class ConversationDebouncer:
    """
    Tras cada mensaje, reinicia la espera. Solo cuando pasan `delay_seconds`
    sin nuevos mensajes para ese `conversation_id` se invoca el callback.
    """

    def __init__(
        self,
        delay_seconds: float = 8.0,
    ) -> None:
        if delay_seconds <= 0:
            raise ValueError("delay_seconds debe ser > 0")
        self.delay_seconds = delay_seconds
        self._lock = asyncio.Lock()
        self._buffers: dict[Hashable, AccumulatedMessage] = {}
        self._tasks: dict[Hashable, asyncio.Task[None]] = {}

    async def push(
        self,
        conversation_id: Hashable,
        text: str | None,
        image_urls: list[str] | None,
        on_flush: OnFlush,
    ) -> None:
        urls = [u for u in (image_urls or []) if u]
        async with self._lock:
            buf = self._buffers.setdefault(conversation_id, AccumulatedMessage())
            if text and text.strip():
                buf.texts.append(text.strip())
            for u in urls:
                if u not in buf.image_urls:
                    buf.image_urls.append(u)

            prev = self._tasks.pop(conversation_id, None)
            if prev is not None and not prev.done():
                prev.cancel()
                try:
                    await prev
                except asyncio.CancelledError:
                    pass

            self._tasks[conversation_id] = asyncio.create_task(
                self._run_after_delay(conversation_id, on_flush)
            )

    async def _run_after_delay(
        self,
        conversation_id: Hashable,
        on_flush: OnFlush,
    ) -> None:
        try:
            await asyncio.sleep(self.delay_seconds)
            async with self._lock:
                buf = self._buffers.pop(conversation_id, None)
                self._tasks.pop(conversation_id, None)
            if buf is None:
                return
            combined = buf.merge_text()
            if not combined and not buf.image_urls:
                return
            await on_flush(conversation_id, buf)
        except asyncio.CancelledError:
            raise
