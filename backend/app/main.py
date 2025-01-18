from fastapi import FastAPI
from app.api.routes import router
import app.config.settings

app = FastAPI(title="Chat with Your Paper")

# Register routes
app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "Healthy"}
