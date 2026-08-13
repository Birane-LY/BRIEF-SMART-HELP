from fastapi import FastAPI
from app.core.config import settings
from app.api.routes.support_ticket import router as support_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=settings.app_name)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permet à n'importe quel site de requêter votre API (Idéal pour le développement)
    allow_credentials=True,
    allow_methods=["*"], # Autorise POST, GET, OPTIONS, etc.
    allow_headers=["*"], # Autorise tous les en-têtes
)

app.include_router(
    support_router, 
    prefix=settings.api_prefix, 
    tags=["Support"]
)


@app.get("/health")
def health_check():
    return {"status": "ok"}