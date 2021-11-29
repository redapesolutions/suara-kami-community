import subprocess as sp
import numpy as np
from sk import SK,get_speech_ts_adaptive
from pdb import set_trace
CHUNK_SIZE = 4000

def video_frames_ffmpeg(url):
    iterator = 0
    cmd = f'ffmpeg -loglevel quiet -re -i {url} -acodec pcm_f32le -f f32le -vn -sn -dn -ar 16000 -ac 1 -'.split(" ")
    p = sp.Popen(cmd, stdout=sp.PIPE)
    asr = SK(model="silero_en_large")
    buffer = []
    current_text = ""
    current_len = 0
    skip = False
    repeated = 0
    while True:
        data = p.stdout.read(CHUNK_SIZE*10)
        data = np.frombuffer(data, dtype=np.float32)
        buffer.append(data)
        lala = np.array(buffer).flatten()
        out = asr.transcribe_array(lala)
        if current_text != out["texts"]:
            current_text = out["texts"]
            print(f"{lala.shape} | {current_text}")
        if  current_len == len(current_text.split(" ")):
            repeated += 1
            if repeated > 8:
                skip = True
        if current_len > 14 or skip:
            buffer = []
            print(current_text)
            current_len = 0
            skip = False
            repeated = 0
            # print("reset")
        else:
            current_len = len(current_text.split(" "))
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