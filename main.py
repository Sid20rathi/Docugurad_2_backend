from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from routes.upload_verify_route import router1
from routes.title_document_verify_route import router2
from routes.model_used_update import router3

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router1,prefix="/files",tags=["Pre uploaded files Verification"])
app.include_router(router2,prefix="/title_document",tags=["Title Document Verification"])
app.include_router(router3,prefix="/model_used",tags=["Which Model is Used"])


@app.get("/")
def health():
    return {"status": "ok","meesage": "Backend is running"}


