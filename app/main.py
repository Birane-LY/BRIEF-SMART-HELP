
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes.support_ticket import router as support_router

app = FastAPI(title=settings.app_name)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://brief-smart-help-front-end.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    support_router,
    prefix=settings.api_prefix,
    tags=["Support"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}
