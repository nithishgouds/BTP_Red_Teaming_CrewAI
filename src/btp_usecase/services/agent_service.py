from __future__ import annotations

from ..crew.crew import run_crew
from ..services.extraction_service import ExtractionService
from ..storage.base import DocumentStorage


class AgentService:
    def __init__(self, storage: DocumentStorage, extraction_service: ExtractionService | None = None):
        self.storage = storage
        self.extraction_service = extraction_service or ExtractionService()

    def _detect_file_type(self, document_id: str, raw: bytes | None = None) -> str:
        if raw is not None:
            if raw.startswith(b"%PDF"):
                return "pdf"
            if raw.startswith(b"PK"):
                return "docx"
            return "txt"

        candidates = [
            (self.storage.base_path / f"{document_id}.txt"),
            (self.storage.base_path / f"{document_id}.pdf"),
            (self.storage.base_path / f"{document_id}.docx"),
            (self.storage.base_path / f"{document_id}.bin"),
        ]
        for path in candidates:
            if path.exists():
                suffix = path.suffix.lower()
                if suffix == ".pdf":
                    return "pdf"
                if suffix == ".docx":
                    return "docx"
                if suffix == ".txt":
                    return "txt"
                return "txt"
        return "txt"

    def run_agent(self, user_query: str, document_id: str | None = None, document_text: str | None = None):
        if document_id:
            raw = self.storage.get(document_id)
            if raw is None:
                raise FileNotFoundError(f"Document not found: {document_id}")
            file_type = self._detect_file_type(document_id, raw)
            extracted = self.extraction_service.extract(file_type, raw)
            document_text = extracted.text or ""

        return run_crew(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
        )
