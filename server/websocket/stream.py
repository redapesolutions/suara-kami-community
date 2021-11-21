import subprocess as sp
import json
import os
import numpy as np
from sk import predict
from pdb import set_trace
import logging
import warnings
# create logger with 'spam_application'
logger = logging.getLogger('transcript')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('transcript.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
CHUNK_SIZE = 4000

def video_frames_ffmpeg(url):
    iterator = 0
    cmd = f'ffmpeg -loglevel quiet -re -i {url} -acodec pcm_f32le -f f32le -vn -sn -dn -ar 16000 -ac 1 -'.split(" ")
    p = sp.Popen(cmd, stdout=sp.PIPE)
    buffer = []
    current_text = ""
    repeated = 0
    while True:
        data = p.stdout.read(CHUNK_SIZE * 40)
        data = np.frombuffer(data, dtype=np.float32)
        buffer.append(data)
        inp = np.array(buffer).flatten()
        print(inp.shape)
        out = predict(inp,model="conformer_small",decoder="v1")
        # logger.info(out)
        print(out["texts"])
        buffer = []
        # if current_text != out["texts"]:
        #     current_text = out["texts"]
        #     print(current_text)
        # else:
        #     repeated += 1
        #     if repeated > 8:
        #         repeated = 0
        #         buffer = []
        #         print("reset")
        # if len(current_text.split(" ")) > 10:
        #     print("reset -2")
        #     buffer = []
        iterator += 1
        if len(data) == 0:
            p.wait()
            print("awaiting")
            #return
        if iterator >= 1000:
            break
        yield data

for i, frame in enumerate(video_frames_ffmpeg("http://localhost:8080/hls/test.m3u8")):
    pass