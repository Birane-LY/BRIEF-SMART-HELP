from fastapi import FastAPI
from app.core.config import settings
from app.api.routes.support_ticket import router as support_router

app = FastAPI(title=settings.app_name)

app.include_router(
    support_router, 
    prefix=settings.api_prefix, 
    tags=["Support"]
)


@app.get("/health")
def health_check():
    return {"status": "ok"}