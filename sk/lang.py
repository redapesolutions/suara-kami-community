# based on code from https://github.com/zafercavdar/fasttext-langdetect

import os
from typing import Dict, Union

import fasttext
import wget

models = {"low_mem": None, "high_mem": None}
FTLANG_CACHE = os.getenv("FTLANG_CACHE", "/tmp/fasttext-langdetect")


def download_model(name):
    url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{name}"
    target_path = os.path.join(FTLANG_CACHE, name)
    if not os.path.exists(target_path):
        os.makedirs(FTLANG_CACHE, exist_ok=True)
        wget.download(url=url, out=target_path)
    return target_path


def get_or_load_model(low_memory=False):
    if low_memory:
        model = models.get("low_mem", None)
        if not model:
            model_path = download_model("lid.176.ftz")
            model = fasttext.load_model(model_path)
            models["low_mem"] = model
        return model
    else:
        model = models.get("high_mem", None)
        if not model:
            model_path = download_model("lid.176.bin")
            model = fasttext.load_model(model_path)
            models["high_mem"] = model
        return model


def detect(text: str,k=1,threshold=0.0,low_memory=False) -> Dict[str, Union[str, float]]:
    model = get_or_load_model(low_memory)
    labels, scores = model.predict(text,k=k,threshold=threshold)
    labels = [i.replace("__label__", '') for i in labels]
    scores = [min(float(i), 1.0) for i in scores]
    return {
        "lang": labels[:k],
        "score": scores[:k],
    }

########################################################################

def is_english(text):
    out = detect(text)
    if "en" in out["lang"]:
        return True
    else:
        return False

def is_malay(text):
    out = detect(text)
    if "id" in out["lang"] or "ms" in out["lang"]:
        return True
    else:
        return False

def is_manglish(text):
    out = detect(text,k=5)
    if "en" in out["lang"] and ("id" in out["lang"] or "ms" in out["lang"]):
        return True
    else:
        return False