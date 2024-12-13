from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    file_content = await file.read()
    file_size = len(file_content)  # File size in bytes
    print(f"Received file: {file.filename}, Size: {file_size} bytes")
    return {"filename": file.filename, "size": file_size}