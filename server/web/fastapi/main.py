from fastapi import FastAPI, File, UploadFile,Body
from fastapi.middleware.cors import CORSMiddleware
from sk import predict
import shutil
from pathlib import Path
from uuid import uuid4
import os
from fastapi.responses import HTMLResponse
# from mongoframes import *
# from pymongo import MongoClient
# dbpass = os.environ.get("dbpass")
# Frame._client = MongoClient(f'mongodb+srv://khursani8:{dbpass}@cluster0-0zend.gcp.mongodb.net/suarakami?retryWrites=true&w=majority')

# class Audio(Frame):
#     _fields = {
#         'id',
#         'label',
#         'predicted',
#         'dest'
#         }
    

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


@app.post("/transcript")
async def transcript(file: UploadFile = File(...),name:str=Body(...),label:str=Body(...)):
    dest = Path(f"files/{uuid4()}")
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    transcript = predict(dest)["texts"][0]
    if name=="hidden":
        name = ""
    if label=="no label":
        label = ""
    print("param",name,label)
    # audio = Audio(name=name,label=label,predicted=transcript,dest=str(dest))
    # audio.insert()
    return transcript

@app.get("/",response_class=HTMLResponse)
def read_root():
    # html_content = open("pages/demo.html").read()
    html_content = "test"
    return HTMLResponse(content=html_content, status_code=200)

predict("./test.wav")