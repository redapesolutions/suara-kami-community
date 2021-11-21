from pathlib import Path,PosixPath
from tqdm import tqdm
import requests
import shutil
import multiprocessing
from pqdm.threads import pqdm
import librosa
import string
import os
from fastcore.basics import patch,setify
import numpy as np
from fastcore.foundation import L

def get_labels(model_name):
    if "en" in model_name:
        labels = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',"'",'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','##r','##a','##y','##f','##i','##s','##h','##d','##e','##p','##o','##l','##c','##n','##m','##t','##g','##w','##v','##u','##x','##b','##k','##z','##q','##j','th','the','##er','##nd','##in','##ed','##ou','##at','##en','and','##or','##es','to','of','##on','##is','##ing','##ar','##it','##as','##an','##ll','in','##re','wh','he','##om','be','ha','##le','##ot','##ow','##ic','##ut','it','##ld','that','sh','##ly','was','##gh','##id','##se','on','##ve','##et','##ent','you','##im','##st','##ce','##ith','for','##ir','##ion','as','##ay','his','##al','with','##ur','st','##ter','her','##ver','##ch','re','##ad','##ght','an']
    else:
        labels = list(string.ascii_lowercase) + [" "]
    return labels + ["_"]

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


def inference(model,xs):
    output = model(xs)
    log_probs = output
    entropy = -(np.exp(log_probs) * log_probs).sum(-1).mean(-1)
    log_probs = log_probs.argmax(-1)
    text = decode(log_probs,get_labels(model._model_path.split("/")[-1]))
    timesteps = [0]
    return text,entropy,timesteps

def inference_lm(model,xs,decoder):
    output = model(xs)
    log_probs = output
    out = decoder.decode_beams(to_numpy(log_probs),prune_history=True)
    text, lm_state, timesteps, logit_score, lm_score = out[0]
    entropy = -(np.exp(log_probs) * log_probs).sum(-1)
    time = [i[-1] for i in timesteps]
    entropy = [entropy[i[0]:i[1]].sum().item() for i in time]
    duration = xs.shape[-1] / 16_000
    mult = duration / log_probs.shape[0]
    tt = []
    for i in timesteps:
        left = i[1][0]*mult
        l = divmod(left,1)
        left = l[0] + (l[1] * 0.06)
        right = i[1][1]*mult
        r = divmod(right,1)
        right = r[0] + (r[1] * 0.06)
        tt.append((i[0], round(left,2),round(right,2) ))
    return text,entropy,tt

def process_text(a):
    return a.replace(".","").replace(",",'').replace("?","").replace("!","")\
                .replace('-'," ").replace('"',"").replace("'",'').replace('–'," ")\
                .replace("\\"," ").replace("/"," ").replace("*","").replace("&","dan").lower()

def read_audio(fn):
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

def decode(out,labels):
    if any(["##" in i for i in labels]):
        from collections import OrderedDict
        special_toks = ['[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]']
        vocabs = OrderedDict([(idx,i) for idx,i in enumerate(labels[:-1])])
        tokens = [vocabs.get(i,vocabs[1]) for i in out]
        tokens = [t for t in tokens if t not in special_toks]
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
    else:
        out2 = ["_"]+list(out)
        collapsed = []
        for idx,i in enumerate(out): # can run in parallel
            if i!=out2[idx] and i!=len(labels)-1:
                collapsed.append(i)
        return "".join([labels[i] for i in collapsed])

def decodes(output):
    n_jobs = 4
    return pqdm(output,decode,n_jobs=n_jobs,disable=True)

cache_model = {}

def load_model(path):
    if not isinstance(path,str):
        return path

    path = Path(path)
    download_map = {
        "conformer_small": ["https://zenodo.org/record/5115792/files/conformer_small.onnx?download=1","https://f001.backblazeb2.com/file/suarakami/conformer_small.onnx"],
        "conformer_tiny": ["https://f001.backblazeb2.com/file/suarakami/conformer_tiny.onnx"],
        "conformer_medium": ["https://zenodo.org/record/5674714/files/conformer_medium.onnx?download=1","https://f001.backblazeb2.com/file/suarakami/conformer_medium.onnx"],
        "conformer_small_en": ["https://zenodo.org/record/5716289/files/stt_en_conformer_ctc_small.onnx?download=1"]
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
    # print("loading model")
    if path.is_dir():
        import tensorflow as tf
        tf_model = tf.saved_model.load(str(path))
        model = tf_model.signatures["serving_default"]
    elif path.suffix == ".onnx":
        sess_options = onnxruntime.SessionOptions()
        # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
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

def load_lm(path):
    # print("loading language model")
    if not isinstance(path,str):
        return path
        
    labels = get_labels(path)

    download_map = {
        "v1":["https://zenodo.org/record/5117101/files/out.trie.zip?download=1"],
        "en":["https://zenodo.org/record/5716345/files/mixed-lower.binary.zip?download=1"]
    }
    from pyctcdecode import build_ctcdecoder

    if path in download_map:
        # download weight
        urls = download_map[path]
        # print("recommend to build it yourself based on README tutorial")
        for url in urls:
            filename = os.path.basename(url).split("?")[0]
            dl_path = os.path.join(Path.home(), ".sk/lm")
            os.makedirs(dl_path, exist_ok=True)
            abs_path = os.path.join(dl_path, filename)
            try:
                target_path = abs_path.replace(".zip",".klm")
                if not (Path(target_path).is_file() or Path(abs_path).is_file()):
                    chunk_size = 1024
                    print(f"downloading {filename} language model of size 600+MB, might take a while")
                    with requests.get(url, stream=True) as r, open(abs_path, "wb") as f:
                        for chunk in tqdm(r.iter_content(chunk_size=chunk_size),total=int(int(r.headers["Content-Length"])/chunk_size)):
                            f.write(chunk)
                abs_path = Path(abs_path)
                if abs_path.is_file():
                    shutil.unpack_archive(abs_path,abs_path.parent)
                    Path(abs_path).unlink() # delete zip file
                    print("saved to:",target_path)
                path = target_path
            except:
                continue
            break

    decoder = build_ctcdecoder(
        labels[:-1]+[""],
        path,
        alpha=0.5,
        beta=1
    )
    return decoder

def create_labelling(labels,wav_splits,output):
    times = [s.start / 16_000 for s in wav_splits]
    labelling = []
    start_time = 0

    for i,time in enumerate(times):
        if i>0 and labels[i]!=labels[i-1]:
            temp = [str(labels[i-1]),start_time,time]
            labelling.append(tuple(temp))
            start_time = time
        if i==len(times)-1:
            temp = [str(labels[i]),start_time,time]
            labelling.append(tuple(temp))
    speaker = []
    for l in labelling:
        start = l[1]
        end = l[2]
        chunks = list(filter(lambda i:i[1]>=start and i[2]<=end,output))
        if len(chunks) == 0:
            continue
        else:
            speaker.append([l[0],chunks])
    return speaker