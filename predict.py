import onnx
import onnxruntime
import torch
from torch import tensor
from utils import labels,read_audio,get_files
from multiprocessing import Pool
from pqdm.threads import pqdm
from pathlib import Path
import numpy as np
from fastcore.basics import patch
from pdb import set_trace
from tqdm import tqdm

blank = labels.blank()
space = labels.space()

def get_seq_len(seq_len):
    # Assuming that center is True is stft_pad_amount = 0
    pad_amount = 512 // 2 * 2
    seq_len = torch.floor((seq_len + pad_amount - 512) / 160) + 1
    return seq_len.to(dtype=torch.long)

try:
    import onnxruntime
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    @patch
    def __call__(self:onnxruntime.capi.onnxruntime_inference_collection.InferenceSession, xs, length=None):
        ort_inputs = {self.get_inputs()[0].name: to_numpy(xs)}
        ologits = self.run(None, ort_inputs)
        alogits = np.asarray(ologits)
        logits = torch.from_numpy(alogits[0])
        return logits
except Exception as e:
    print(e)

def decode(out):
    # length = length.int()
    # out = out[:length]#.cpu().numpy()
    out2 = ["_"]+list(out)
    collapsed = []
    for idx,i in enumerate(out):
        if i!=out2[idx] and i!=blank:
            collapsed.append(i)
    return "".join([labels.chars[i] for i in collapsed])

def decodes(output):
    if breakpoint:
        n_jobs = 1
    else:
        n_jobs = 4
    return pqdm(output,decode,n_jobs=n_jobs,disable=True)

def inference(model,xs,xn=None):
    output = model(xs,xn)
    log_probs = torch.nn.functional.log_softmax(output,-1) # for entropy later
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean(-1)
    log_probs = log_probs.argmax(2).type(torch.int)
    log_probs = log_probs.contiguous()
    text = decode(log_probs[0])
    timesteps = [0]
    return text,entropy,timesteps

def predict(model,fn):
    """Predicting speech to text

    Args:
        model ([type]): [Pytorch or onnx model]
        fn (path): [audio filepath]
        normalize (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if isinstance(model,str):
        model = onnx.load(model)
        model = onnxruntime.InferenceSession(model.SerializeToString())
    if isinstance(fn,str):
        fn = [fn]
    if Path(fn[0]).is_file():
        preds = []
        ents = []
        for i in tqdm(fn):
            data,_ = read_audio(str(i))
            xs = torch.as_tensor(data)[None]
            text,entropy,timesteps = inference(model,xs)
            preds.append(text)
            ents.append(entropy)
        return preds,fn,entropy
    else:
        files = []
        for i in fn:
            files.extend(get_files(i,[".wav"],recurse=True))
        preds = []
        ents = []
        for i in tqdm(files,total=len(files)):
            data,sr = read_audio(str(i))
            xs = torch.as_tensor(data)[None]
            text,entropy,timesteps = inference(model,xs)
            preds.append(text)
            ents.append(entropy)
        return preds,files,ents

if __name__ == "__main__":
    import fire
    fire.Fire(predict)
    # text = predict()