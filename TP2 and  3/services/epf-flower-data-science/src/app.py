from fastapi import FastAPI, Request, HTTPException, Depends
from starlette.middleware.cors import CORSMiddleware
from firebase_admin import auth
from fastapi.security import HTTPBearer

from src.api.router import router

security = HTTPBearer()

def get_application() -> FastAPI:
    application = FastAPI(
        title="epf-flower-data-science",
        description="""Fast API""",
        version="1.0.0",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application
