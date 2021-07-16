import onnx
import onnxruntime
import torch
from torch import tensor
from helper import read_audio,get_files
from multiprocessing import Pool
from pqdm.threads import pqdm
from pathlib import Path
import numpy as np
from fastcore.basics import patch
from pdb import set_trace
from tqdm import tqdm
import string

labels =  list(
    string.ascii_lowercase  # + string.digits
) + [" ","_"]

blank = labels.index("_")

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
        alogits = np.asarray(ologits[0])
        logits = torch.from_numpy(alogits[0])
        return logits,int(ologits[1][0])
except Exception as e:
    print(e)

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
    output,ln = model(xs)
    log_probs = torch.nn.functional.log_softmax(output,-1) # for entropy later
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean(-1)
    log_probs = log_probs.argmax(-1).type(torch.int)
    log_probs = log_probs.contiguous()
    text = decode(log_probs)
    timesteps = [0]
    return text,entropy,timesteps

def inference_lm(model,xs,decoder):
    output,ln = model(xs)
    if output.ndim == 3:
        output = output[0]
        ln = ln[0]
    output = output[:ln]
    probs = torch.nn.functional.softmax(output,-1)
    out = decoder.decode_beams(to_numpy(probs),prune_history=True)
    text, lm_state, timesteps, logit_score, lm_score = out[0]
    log_probs = probs.log()
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    time = [i[-1] for i in timesteps]
    entropy = [entropy[i[0]:i[1]].sum().item() for i in time]
    return text,entropy,timesteps

def predict(model,fn,decoder=None):
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
            if decoder:
                text,entropy,timesteps = inference_lm(model,xs,decoder)
            else:
                text,entropy,timesteps = inference(model,xs)
            preds.append(text)
            ents.append(entropy)
        return preds,fn,ents,timesteps
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
        return preds,files,ents,timesteps

if __name__ == "__main__":
    import fire
    fire.Fire(predict)
    # text = predict()