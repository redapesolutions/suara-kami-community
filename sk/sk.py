#! /usr/bin/env python
import sys
from pathlib import Path,PosixPath
from pdb import set_trace

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from fastcore.foundation import L
from tqdm import tqdm
import subprocess

import os
from smart_open.smart_open_lib import patch_pathlib
patch_pathlib()
import numpy as np
from scipy.io.wavfile import read
import io
import warnings
from sk.utils import *

info_path = Path.home()/".sk/info.txt"

def transcribe_file(files,model,decoder=None,logits=False,speaker=False,verbose=True):
    preds = []
    ents = []
    timestamps = []
    processed_files = []
    all_logits = []
    speakers = []
    if speaker:
        try:
            from resemblyzer import preprocess_wav, VoiceEncoder
            from spectralcluster import SpectralClusterer
        except:
            print("installing resemblyzer")
            os.system("pip install resemblyzer --no-deps")
            os.system("pip install spectralcluster==0.1.0")
            from resemblyzer import preprocess_wav, VoiceEncoder
            from spectralcluster import SpectralClusterer
        encoder = VoiceEncoder("cpu")
        clusterer = SpectralClusterer(
            min_clusters=1,
            max_clusters=5,
            p_percentile=0.95,
            gaussian_blur_sigma=30
        )
    # print("start prediction")
    for i in tqdm(files,total=len(files),disable=not verbose):
        spkr = ["not enabled"]
        try:
            data,_ = read_audio(str(i))
        except Exception as e:
            print(f"failed,",e)
            continue
        xs = data[None]
        if logits:
            all_logits.append(model(xs))
            processed_files.append(i)
            continue
        if decoder:
            if isinstance(decoder,str):
                decoder = load_lm(decoder)
            text,entropy,tt = inference_lm(model,xs,decoder)
            if speaker:
                _, cont_embeds, wav_splits = encoder.embed_utterance(preprocess_wav(str(i)), return_partials=True, rate=16)
                labels = clusterer.predict(cont_embeds)
                spkr = create_labelling(labels,wav_splits,tt)
        else:
            text,entropy,tt = inference(model,xs)

        preds.append(text)
        ents.append(entropy)
        timestamps.append(tt)
        processed_files.append(i)
        speakers.append(spkr)
    return preds,ents,timestamps,processed_files,speakers,all_logits

def transcribe_bytes(b,model,decoder=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _,xs = read(io.BytesIO(b))
    xs = xs.astype(np.float32)[None]
    if decoder:
        text,entropy,tt = inference_lm(model,xs,decoder)
    else:
        text,entropy,tt = inference(model,xs)
    return {
        "texts":text,
        "filenames":len(b),
        "entropy":entropy,
        "timestamps":tt,
    }

def transcribe_array(array,model,decoder=None):
    array = np.pad(array,(0,268800-array.shape[0]))
    xs = array[None]
    if decoder:
        text,entropy,tt = inference_lm(model,xs,decoder)
    else:
        text,entropy,tt = inference(model,xs)
    return {
        "texts":text,
        "filenames":array[:16],
        "entropy":entropy,
        "timestamps":tt,
    }

def predict(fn,model="conformer_small",decoder=None,output_folder=None,output_csv=None,audio_type=".wav",logits=False,speaker=False,verbose=True):
    """Predicting speech to text

    Args:
        model ([type]): [Pytorch or onnx model]
        fn (path): [audio filepath]
        normalize (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if output_csv:
        try:
            import pandas as pd
        except Exception as e:
            print(e)
            return

    if isinstance(model,str):
        model = load_model(model)
        if model is None:
            print("loading model failed")
            return {
                "texts":[],
                "filenames":[],
                "entropy":[],
                "timestamps":[]
            }

    if __name__ == "__main__" and decoder is None:
        print("Can add '--decoder v1' to improve accuracy or prepare your own language model based on README")

    if speaker and decoder is None:
        print("enable decoder v1 for speaker timesteps")
        decoder = "v1"

    if isinstance(decoder,str):
        decoder = load_lm(decoder)

    if isinstance(fn,bytes):
        return transcribe_bytes(fn,model,decoder)
    if isinstance(fn,np.ndarray):
        return transcribe_array(fn,model,decoder)

    if isinstance(fn,PosixPath):
        fn = str(fn)

    if isinstance(fn,str):
        fn = fn.split(",") # string to list, might break for filename with comma

    files = []
    if verbose:
        print("Total input path:",len(fn))
    for i in fn:
        files.extend([i] if Path(i).is_file() else get_files(i,audio_type.split(","),recurse=True))
    if verbose:
        print(f"Total audio found({audio_type}):",len(files))

    preds,ents,timestamps,processed_files,speakers,all_logits = transcribe_file(files,model,decoder,logits,speaker,verbose)

    if __name__ != "__main__":
        fn = [Path(i) for i in processed_files]

    if logits:
        return {"logits":all_logits,"filenames":fn}

    if output_folder:
        output = Path(output_folder)
        output.mkdir(exist_ok=True)
        print("saving prediction to:",str(output))
        for p,pred in zip(fn,preds):
            with open(output/f"{p.name}_sk.txt","w+") as f:
                f.write(pred)

    if output_csv:
        df = pd.DataFrame([fn,preds,ents,timestamps])
        df.columns = ["filename","prediction","entropy","timestamps"]
        if ".csv" not in output_csv:
            output_csv += ".csv"
        df.to_csv(output_csv)
        print("write csv to:",output_csv)

    if __name__ == "__main__" and info_path.is_file() and int(info_path.read_text()) > 0:
        print(f"Saving {len(fn)} data")
        [subprocess.Popen(f"feedback {i}",shell=True) for i in tqdm(fn)]

    if output_folder or output_csv:
        return f"done -> {output_folder or output_csv}"

    return {
        "texts":preds,
        "filenames":fn,
        "entropy":ents,
        "timestamps":timestamps,
        "speakers":speakers
    }

if __name__ == "__main__":
    if info_path.is_file():
        info = info_path.read_text()
    else:
        msg = "Are you agree to allow us anonymously collecting your data for improving our model performance? (y/n): "
        answer = input(msg)
        info_path.parent.mkdir(exist_ok=True)
        with info_path.open("w+") as f:
            if answer == "y":
                answer2 = input("Are you agree to share the collected data to the community? (y/n) :")
                if answer2 == "y":
                    f.write("2")
                else:
                    f.write('1')
            else:
                f.write('0')
    import fire
    fire.Fire(predict)
