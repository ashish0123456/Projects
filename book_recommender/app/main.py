from fastapi import FastAPI
from app.routes import recommend

app = FastAPI(title="Book Recommender API")
app.include_router(recommend.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
