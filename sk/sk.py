#! /usr/bin/env python
import onnxruntime
import torch
from pqdm.threads import pqdm
from pathlib import Path
import numpy as np
from fastcore.basics import patch,setify
from fastcore.foundation import L
from pdb import set_trace
from tqdm import tqdm
import string
import multiprocessing

import soundfile as sf
import librosa
import os
from pathlib import Path
import numpy as np

labels =  list(
    string.ascii_lowercase  # + string.digits
) + [" ","_"]

blank = labels.index("_")

def read_audio(fn):
    with sf.SoundFile(fn, 'r') as f:
        sample_rate = f.samplerate
        samples = f.read(dtype="float32").transpose()
        if 16000 != sample_rate:
            samples = librosa.core.resample(samples, sample_rate, 16000)
            sample_rate = 16000
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
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    @patch
    def __call__(self:onnxruntime.capi.onnxruntime_inference_collection.InferenceSession, xs, length=None):
        ort_inputs = {self.get_inputs()[0].name: to_numpy(xs)}
        ologits = self.run(None, ort_inputs)
        alogits = np.asarray(ologits[0])
        logits = torch.from_numpy(alogits[0])
        return logits,int(ologits[1][0])

except Exception as e:
    print(e)
    print("onnx not supported")

def decode(out):
    out2 = ["_"]+list(out)
    collapsed = []
    for idx,i in enumerate(out): # can run in parallel
        if i!=out2[idx] and i!=blank:
            collapsed.append(i)
    return "".join([labels[i] for i in collapsed])

def decodes(output):
    if breakpoint:
        n_jobs = 1
    else:
        n_jobs = 4
    return pqdm(output,decode,n_jobs=n_jobs,disable=True)

def inference(model,xs):
    if "tensorflow" in str(type(model)):
        import tensorflow as tf
        xs = tf.constant(xs)
        output = model(xs)["output_0"].numpy()
        output = torch.as_tensor(output)[0]
    else:
        output,_ = model(xs)
    log_probs = torch.nn.functional.log_softmax(output,-1) # for entropy later
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean(-1)
    log_probs = log_probs.argmax(-1).type(torch.int)
    log_probs = log_probs.contiguous()
    text = decode(log_probs)
    timesteps = [0]
    return text,entropy,timesteps

def inference_lm(model,xs,decoder):
    if "tensorflow" in str(type(model)):
        import tensorflow as tf
        xs = tf.constant(xs)
        output = model(xs)["output_0"].numpy()
        output = torch.as_tensor(output)[0]
    else:
        output,_ = model(xs)
    if output.ndim == 3:
        output = output[0]
    probs = torch.nn.functional.softmax(output,-1)
    out = decoder.decode_beams(to_numpy(probs),prune_history=True)
    text, lm_state, timesteps, logit_score, lm_score = out[0]
    log_probs = probs.log()
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    time = [i[-1] for i in timesteps]
    entropy = [entropy[i[0]:i[1]].sum().item() for i in time]
    return text,entropy,timesteps

def load_model(path):
    download_map = {
        "conformer_small": "",
        "conformer_tiny": "",
    }
    if path in download_map:
        # download weight
        url = download_map[path]
        model_path = Path("~/.sk/models").mkdir(exist_ok=True)
        
    path = Path(path)
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
        model = onnxruntime.InferenceSession(str(path),sess_options)
    elif path.suffix == ".pth":
        model = torch.load(str(path))
        model = model.eval()
    else:
        print("model type not recognized")
        return path
    return model

def load_decoder(path):
    from pyctcdecode import build_ctcdecoder
    import kenlm
    kenlm_model = kenlm.Model(path)

    decoder = build_ctcdecoder(
        labels,
        kenlm_model, 
        alpha=0.5,
        beta=1.0, 
        ctc_token_idx=labels.index("_")
    )

    return decoder

def predict(model,fn,decoder=None,output_folder=None):
    """Predicting speech to text

    Args:
        model ([type]): [Pytorch or onnx model]
        fn (path): [audio filepath]
        normalize (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if isinstance(model,str):
        # model = load_model(model)
        import multiprocessing
        sess_options = onnxruntime.SessionOptions()
        sess_options = onnxruntime.SessionOptions()
        # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        model = onnxruntime.InferenceSession(model,sess_options)

    if isinstance(decoder,str):
        decoder = load_decoder(decoder)

    if isinstance(fn,str):
        fn = [fn]
        
    if Path(fn[0]).is_file():
        preds = []
        ents = []
        for i in tqdm(fn):
            data,_ = read_audio(str(i))
            xs = torch.as_tensor(data)[None]
            if decoder:
                text,entropy,timesteps = inference_lm(model,xs,decoder)
            else:
                text,entropy,timesteps = inference(model,xs)
            preds.append(text)
            ents.append(entropy)
    else:
        files = []
        for i in fn:
            files.extend(get_files(i,[".wav"],recurse=True))
        preds = []
        ents = []
        for i in tqdm(files,total=len(files)):
            data,_ = read_audio(str(i))
            xs = torch.as_tensor(data)[None]
            if decoder:
                text,entropy,timesteps = inference_lm(model,xs,decoder)
            else:
                text,entropy,timesteps = inference(model,xs)
            preds.append(text)
            ents.append(entropy)
        fn = files

    fn = [Path(i) for i in fn]

    if output_folder:
        output = Path(output_folder)
        output.mkdir(exist_ok=True)
        for p,pred in zip(fn,preds):
            with open(output/f"{p.name}_sk.txt","w+") as f:
                f.write(pred)

        return "done"

    return preds,fn,ents,timesteps
    

if __name__ == "__main__":
    import fire
    fire.Fire(predict)