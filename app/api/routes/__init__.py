"""
API router aggregation — includes all endpoint routers.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.chat import router as chat_router
from app.api.routes.research import router as research_router
from app.api.routes.documents import router as documents_router
from app.api.routes.contracts import router as contracts_router
from app.api.routes.cases import router as cases_router
from app.api.routes.drafting import router as drafting_router
from app.api.routes.citations import router as citations_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.runs import router as runs_router
from app.api.routes.human_review import router as hr_router
from app.api.routes.admin import router as admin_router

api_router = APIRouter()

# Core endpoints
api_router.include_router(health_router, tags=["health"])
api_router.include_router(chat_router, tags=["chat"])
api_router.include_router(research_router, tags=["research"])
api_router.include_router(documents_router, tags=["documents"])
api_router.include_router(contracts_router, tags=["contracts"])
api_router.include_router(cases_router, tags=["cases"])
api_router.include_router(drafting_router, tags=["drafting"])
api_router.include_router(citations_router, tags=["citations"])
api_router.include_router(retrieval_router, tags=["retrieval"])
api_router.include_router(runs_router, tags=["runs"])
api_router.include_router(hr_router, tags=["human-review"])

# Admin endpoints (with /admin prefix handled in the admin router)
api_router.include_router(admin_router, tags=["admin"])

__all__ = ["api_router"]
