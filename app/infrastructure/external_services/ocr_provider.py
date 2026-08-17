"""
OCR provider abstraction and Tesseract implementation.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.enums import OCRProviderType
from app.domain.interfaces import OCRProvider

logger = get_logger()


class TesseractOCRProvider(OCRProvider):
    """OCR provider using Tesseract via pytesseract."""

    def __init__(
        self,
        language: str = "eng",
        tesseract_cmd: str | None = None,
        poppler_path: str | None = None,
    ) -> None:
        self._language = language
        self._tesseract_cmd = tesseract_cmd or settings.TESSERACT_CMD
        self._poppler_path = poppler_path or settings.POPPLER_PATH

    @property
    def provider_type(self) -> OCRProviderType:
        return OCRProviderType.TESSERACT

    async def extract_text(
        self,
        file_path: str,
        language: str | None = None,
        dpi: int = 300,
        **kwargs: Any,
    ) -> str:
        import pytesseract
        from pdf2image import convert_from_path, pdfinfo_from_path

        lang = language or self._language

        # Check if it's a PDF
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            # Convert PDF pages to images, then OCR each
            poppler_path = self._poppler_path or (shutil.which("pdfinfo") and os.path.dirname(shutil.which("pdftoppm") or ""))
            if self._poppler_path:
                images = convert_from_path(file_path, dpi=dpi, poppler_path=self._poppler_path)
            else:
                images = convert_from_path(file_path, dpi=dpi)

            texts: list[str] = []
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang=lang)
                texts.append(f"--- Page {i + 1} ---\n{text}")
            return "\n".join(texts)

        # Single image file
        import pytesseract

        text = await asyncio_to_thread(pytesseract.image_to_string, file_path, lang=lang)
        return text

    async def close(self) -> None:
        pass


class OCRMyPDFProvider(OCRProvider):
    """OCR provider using ocrmypdf (best quality for PDF documents)."""

    def __init__(self, language: str = "eng", tesseract_cmd: str | None = None) -> None:
        self._language = language
        self._tesseract_cmd = tesseract_cmd or settings.TESSERACT_CMD

    @property
    def provider_type(self) -> OCRProviderType:
        return OCRProviderType.OCRMYPDF

    async def extract_text(
        self,
        file_path: str,
        language: str | None = None,
        dpi: int = 300,
        **kwargs: Any,
    ) -> str:
        import ocrmypdf
        import pdfplumber

        lang = language or self._language
        suffix = f"_ocr_{os.path.basename(file_path)}"
        output_path = os.path.join(tempfile.gettempdir(), suffix)

        try:
            await asyncio_to_thread(
                ocrmypdf.ocr,
                file_path,
                output_path,
                language=lang,
                force_render_pages=True,
                skip_text=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ocrmypdf_failed", error=str(exc))
            # Fall back to tesseract
            tesseract = TesseractOCRProvider(language=lang, tesseract_cmd=self._tesseract_cmd)
            return await tesseract.extract_text(file_path, language=lang, dpi=dpi)

        # Extract text from OCR'd PDF
        with pdfplumber.open(output_path) as pdf:
            texts = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(texts)

    async def close(self) -> None:
        pass


class NoOpOCRProvider(OCRProvider):
    """No-op OCR provider for when OCR is disabled."""

    @property
    def provider_type(self) -> OCRProviderType:
        return OCRProviderType.NONE

    async def extract_text(
        self,
        file_path: str,
        language: str = "eng",
        dpi: int = 300,
        **kwargs: Any,
    ) -> str:
        return ""

    async def close(self) -> None:
        pass


class OCRProviderFactory:
    """Factory for creating OCR providers."""

    _provider_map = {
        OCRProviderType.TESSERACT: TesseractOCRProvider,
        OCRProviderType.OCRMYPDF: OCRMyPDFProvider,
        OCRProviderType.NONE: NoOpOCRProvider,
    }

    def create(self, provider_type: str | OCRProviderType, **kwargs: Any) -> OCRProvider:
        if isinstance(provider_type, str):
            try:
                provider_type = OCRProviderType(provider_type)
            except ValueError:
                provider_type = OCRProviderType.NONE

        cls = self._provider_map.get(provider_type)
        if cls is None:
            return NoOpOCRProvider()
        if issubclass(cls, TesseractOCRProvider):
            return TesseractOCRProvider(
                language=settings.OCR_LANGUAGE,
                tesseract_cmd=settings.TESSERACT_CMD,
                poppler_path=settings.POPPLER_PATH,
            )
        return cls(**kwargs)


async def asyncio_to_thread(func, *args, **kwargs):
    """Wrapper for asyncio.to_thread."""
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


__all__ = [
    "TesseractOCRProvider",
    "OCRMyPDFProvider",
    "NoOpOCRProvider",
    "OCRProviderFactory",
]
