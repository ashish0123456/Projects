from fastapi import FastAPI
from app.routes import generate

app = FastAPI(title="Transformer Model API")
app.include_router(generate.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
