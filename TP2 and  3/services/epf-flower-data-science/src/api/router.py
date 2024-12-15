"""API Router for Fast API."""
from fastapi import APIRouter

from src.api.routes import hello
from src.api.routes import docs

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])
router.include_router(docs.router, tags=["docs"])
