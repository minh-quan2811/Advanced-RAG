import sys
import os
from app.utils.environment_setup import EnvironmentSetup
from fastapi import FastAPI


app = FastAPI()
env = EnvironmentSetup()
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from app.controllers import upload_controller
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles



# app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(upload_controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
