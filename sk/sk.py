#! /usr/bin/env python
import onnxruntime
from pqdm.threads import pqdm
import numpy as np
from fastcore.basics import patch,setify
from fastcore.foundation import L
from tqdm import tqdm
import string
import multiprocessing
import subprocess

import soundfile as sf
import librosa
import os
from pathlib import Path,PosixPath
from smart_open.smart_open_lib import patch_pathlib
patch_pathlib()
import numpy as np
import requests
from scipy.io.wavfile import read
import io
import warnings

info_path = Path.home()/".sk/info.txt"

labels =  list(
    string.ascii_lowercase  # + string.digits
) + [" ","_"]

blank = labels.index("_")

def process_text(a):
    return a.replace(".","").replace(",",'').replace("?","").replace("!","")\
                .replace('-'," ").replace('"',"").replace("'",'').replace('–'," ")\
                .replace("\\"," ").replace("/"," ").replace("*","").replace("&","dan").lower()

def read_audio(fn):
    try:
        with sf.SoundFile(fn, 'r') as f:
            sample_rate = f.samplerate
            samples = f.read(dtype="float32").transpose()
            if 16000 != sample_rate:
                samples = librosa.core.resample(samples, sample_rate, 16000)
                sample_rate = 16000
            if samples.ndim > 2:
                samples = np.mean(samples,1)
    except:
        print("warning: Not wav file, using slow audio reader")
        samples,sample_rate = librosa.load(fn,16_000)
        if samples.ndim > 2:
                samples = np.mean(samples,1)
    return samples,sample_rate

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders=L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)

try:
    import onnxruntime
    def to_numpy(tensor):
        return tensor
        if "requires_grad" in tensor:
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        else:
            return np.array(tensor)
    @patch
    def __call__(self:onnxruntime.capi.onnxruntime_inference_collection.InferenceSession, xs, length=None):
        ort_inputs = {self.get_inputs()[0].name: to_numpy(xs)}
        ologits = self.run(None, ort_inputs)
        alogits = np.asarray(ologits[0])
        logits = alogits[0]
        return logits

except Exception as e:
    print(e)
    print("onnx not supported")

cache_model = {}

def load_model(path):
    path = Path(path)

    download_map = {
        "conformer_small": ["https://zenodo.org/record/5115792/files/conformer_small.onnx?download=1","https://f001.backblazeb2.com/file/suarakami/conformer_small.onnx"],
        "conformer_tiny": ["https://f001.backblazeb2.com/file/suarakami/conformer_tiny.onnx"],
    }
    if path.stem in cache_model:
        return cache_model[path.stem]
    path = str(path)
    if path in download_map:
        # download weight
        urls = download_map[path]
        for url in urls:
            filename = os.path.basename(url).split("?")[0]
            dl_path = os.path.join(Path.home(), ".sk/models")
            os.makedirs(dl_path, exist_ok=True)
            abs_path = os.path.join(dl_path, filename)
            abs_path = Path(abs_path)
            try:
                if not abs_path.is_file():
                    chunk_size = 1024
                    print("downloading:",filename)
                    with requests.get(url, stream=True) as r, open(abs_path, "wb") as f:
                        for chunk in tqdm(r.iter_content(chunk_size=chunk_size),total=int(int(r.headers["Content-Length"])/chunk_size)):
                            f.write(chunk)
                    print("saved to:",abs_path)
                path = abs_path
            except:
                continue
            break

    path = Path(path)
    print("loading model")
    if path.is_dir():
        import tensorflow as tf
        tf_model = tf.saved_model.load(str(path))
        model = tf_model.signatures["serving_default"]
    elif path.suffix == ".onnx":
        sess_options = onnxruntime.SessionOptions()
        # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        try:
            model = onnxruntime.InferenceSession(str(path),sess_options)
        except Exception as e:
            print(e)
            print(f"onnx model corrupted, please remove {path} and rerun the code again")
            return None

    elif path.suffix == ".pth":
        import torch
        model = torch.load(str(path))
        model = model.eval()
    else:
        print("model type not recognized")
        return path
    cache_model[path.stem] = model
    return model

def load_decoder(path):
    print("loading language model")
    download_map = {
        "v1":["https://zenodo.org/record/5117101/files/out.trie.zip?download=1"]
    }
    from pyctcdecode import build_ctcdecoder
    import kenlm

    if path in download_map:
        # download weight
        urls = download_map[path]
        print("downloading language model of size 600+MB, might take a while")
        print("recommend to build it yourself based on README tutorial")
        for url in urls:
            filename = os.path.basename(url).split("?")[0]
            dl_path = os.path.join(Path.home(), ".sk/lm")
            os.makedirs(dl_path, exist_ok=True)
            abs_path = os.path.join(dl_path, filename)
            try:
                target_path = abs_path.replace(".zip","")
                if not (Path(target_path).is_file() or Path(abs_path).is_file()):
                    chunk_size = 1024
                    print("downloading:",filename)
                    with requests.get(url, stream=True) as r, open(abs_path, "wb") as f:
                        for chunk in tqdm(r.iter_content(chunk_size=chunk_size),total=int(int(r.headers["Content-Length"])/chunk_size)):
                            f.write(chunk)
                if Path(abs_path).is_file():
                    with zipfile.ZipFile(abs_path,"r") as zip_ref:
                        target_path = abs_path.replace(".zip","")
                        zip_ref.extractall(target_path)
                    Path(abs_path).unlink() # delete zip file
                path = target_path
                print("saved to:",path)
            except:
                continue
            break

    kenlm_model = kenlm.Model(path)

    decoder = build_ctcdecoder(
        labels,
        kenlm_model, 
        alpha=0.5,
        beta=1.0, 
        ctc_token_idx=labels.index("_")
    )
    return decoder

def decode(out):
    out2 = ["_"]+list(out)
    collapsed = []
    for idx,i in enumerate(out): # can run in parallel
        if i!=out2[idx] and i!=blank:
            collapsed.append(i)
    return "".join([labels[i] for i in collapsed])

def decodes(output):
    n_jobs = 4
    return pqdm(output,decode,n_jobs=n_jobs,disable=True)

def inference(model,xs):
    if "tensorflow" in str(type(model)):
        import tensorflow as tf
        xs = tf.constant(xs)
        output = model(xs)["output_0"].numpy()
    else:
        output = model(xs)
    log_probs = output
    entropy = -(np.exp(log_probs) * log_probs).sum(-1).mean(-1)
    log_probs = log_probs.argmax(-1)
    text = decode(log_probs)
    timesteps = [0]
    return text,entropy,timesteps

def inference_lm(model,xs,decoder):
    if "tensorflow" in str(type(model)):
        import tensorflow as tf
        xs = tf.constant(xs)
        output = model(xs)["output_0"].numpy()
    else:
        output = model(xs)
    log_probs = output
    out = decoder.decode_beams(to_numpy(log_probs),prune_history=True)
    text, lm_state, timesteps, logit_score, lm_score = out[0]
    entropy = -(np.exp(log_probs) * log_probs).sum(-1)
    time = [i[-1] for i in timesteps]
    entropy = [entropy[i[0]:i[1]].sum().item() for i in time]
    return text,entropy,timesteps

def predict(fn,model=None,decoder=None,output_folder=None,output_csv=None,audio_type=".wav",logits=False):
    """Predicting speech to text

    Args:
        model ([type]): [Pytorch or onnx model]
        fn (path): [audio filepath]
        normalize (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    global data
    if model is None: # if model not defined,use default model
        model = "conformer_small"
    if isinstance(model,str):
        model = load_model(model)
        if model is None:
            print("loading model failed")
            return {
                "texts":[],
                "filenames":[],
                "entropy":[],
                "timesteps":[]
            }
    if isinstance(decoder,str):
        decoder = load_decoder(decoder)

    if output_csv:
        try:
            import pandas as pd
        except Exception as e:
            print(e)
            return

    if isinstance(fn,PosixPath):
        fn = str(fn)

    if isinstance(fn,str):
        fn = fn.split(",") # string to list, might break for filename with comma
    
    if isinstance(fn,bytes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _,xs = read(io.BytesIO(fn))
        xs = xs.astype(np.float32)[None]
        if decoder:
            text,entropy,tt = inference_lm(model,xs,decoder)
        else:
            text,entropy,tt = inference(model,xs)
        return {
            "texts":text,
            "filenames":fn,
            "entropy":entropy,
            "timesteps":tt,
        }
            
    files = []
    print("Total input path:",len(fn))
    for i in fn:
        files.extend([i] if Path(i).is_file() else get_files(i,audio_type.split(","),recurse=True))
    print(f"Total audio found({audio_type}):",len(files))

    preds = []
    ents = []
    timesteps = []
    processed_files = []
    all_logits = []
    print("start prediction")
    for i in tqdm(files,total=len(files)):
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
                decoder = load_decoder(decoder)
            text,entropy,tt = inference_lm(model,xs,decoder)
        else:
            text,entropy,tt = inference(model,xs)
        preds.append(text)
        ents.append(entropy)
        timesteps.append(tt)
        processed_files.append(i)

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
        df = pd.DataFrame([fn,preds,ents,timesteps])
        df.columns = ["filename","prediction","entropy","timesteps"]
        if ".csv" not in output_csv:
            output_csv += ".csv"
        df.to_csv(output_csv)
        print("write csv to:",output_csv)

    if __name__ == "__main__" and info_path.is_file() and int(info_path.read_text()) > 0:
        print(f"Saving {len(fn)} data")
        [subprocess.Popen(f"feedback {i}",shell=True) for i in tqdm(fn)]

    if output_folder or output_csv:
        return "done"

    return {
        "texts":preds,
        "filenames":fn,
        "entropy":ents,
        "timesteps":timesteps,
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