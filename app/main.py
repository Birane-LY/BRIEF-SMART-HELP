from fastapi import FastAPI

app = FastAPI(
    title="Smart Help API",
    description="API d'analyse intelligente des réclamations e-commerce",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}