from fastapi import FastAPI
from app.routes import predict
import uvicorn

app = FastAPI(title='Image Captioning API')

app.include_router(predict.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)