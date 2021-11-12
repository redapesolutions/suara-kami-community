from fastapi import FastAPI, File, UploadFile,Body
from fastapi.middleware.cors import CORSMiddleware
from sk import predict
import shutil
from pathlib import Path
from uuid import uuid4
import os
from fastapi.responses import HTMLResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/files/")
async def get_file_size(file: bytes = File(...)):
    return {"file_size": len(file)}


saved = Path("files")
saved.mkdir(exist_ok=True)

@app.post("/transcript")
# async def transcript(file: UploadFile = File(...),name:str=Body(...),label:str=Body(...)):
async def transcript(file: UploadFile = File(...)):
    dest = saved/f"{uuid4()}"
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    transcript = predict(dest,verbose=False)["texts"][0]
    return transcript

@app.post("/speaker_transcript")
# async def transcript(file: UploadFile = File(...),name:str=Body(...),label:str=Body(...)):
async def speaker_transcript(file: UploadFile = File(...)):
    dest = saved/f"{uuid4()}"
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    transcript = predict(dest,verbose=False,speaker=True,decoder="v1")["texts"][0]
    return transcript

@app.get("/",response_class=HTMLResponse)
def read_root():
    # html_content = open("pages/demo.html").read()
    html_content = "test"
    return HTMLResponse(content=html_content, status_code=200)

# print(predict("./test.wav",verbose=False,speaker=True,decoder="v1"))
print("warming up prediction")
print(predict("./test.wav",verbose=False))