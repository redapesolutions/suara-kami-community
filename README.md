## suara-kami-community

Suara Kami: pre-trained STT models for Bahasa Malaysia.

A simple pipeline for doing speech processing

Works for GPU and CPU

Small and fast

Without pytorch or tf dependencies

support multiple audio type

Can run from CLI and python import

Currently can use our Malay STT model and English model(Manglish/Singlish coming soon)

[Load test report](https://htmlpreview.github.io/?https://github.com/redapesolutions/suara-kami-community/blob/main/loadtest.html)

[Demo](https://huggingface.co/spaces/malay-huggingface/suarakami)

1. Setup

```
pip install git+https://github.com/redapesolutions/suara-kami-community
or
git clone https://github.com/redapesolutions/suara-kami-community
cd suara-kami-community
pip install . --upgrade

fixing error(optional)
1. error: command 'gcc' failed: No such file or directory
-> sudo apt install build-essential gcc
2. OSError: sndfile library not found
-> sudo apt-get install libsndfile1
```

Speech models(ONNX)
- Malay
  1. "conformer_tiny"
  1. "conformer_small"

- English
  1. "silero_en"
  1. "nemo_en"

- Manglish
  1. "conformer_medium"

- Vad
  1. "silero_vad"

### Share data
```
Usage: feedback PATH

For detailed information on this command, run:
  feedback --help
```

```
feedback data_to_share # folder structure should be audio and txt file with same name but different ext for example audio.wav and audio.txt in same folder
feedback data_to_share.zip # same as above
feedback audio.wav
```

GPU Usage

```
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu --upgrade
```

GRPC Server/Client

check [server/grpc](./server/grpc) folder

Web Example

check [server/web](./server/web) folder

Websocket/Streaming Example

check [server/websocket](./server/websocket) folder

3. Tutorials

- [Simple Speech to Text](./tutorials/1.speech_to_text.ipynb)
- [Speech to Text w/ Language model](./tutorials/2.speech_to_text_with_language_model.ipynb)
- [Convert Speech text to Written text form](./tutorials/3.normalize_text.ipynb)
- [Simple Speech to Text using tf saved model](./tutorials/4.tensorflow_inference.ipynb)
- [Diarization](./tutorials/5.diarization.ipynb)
- [Voice Activity Detection](./tutorials/6.Voice_activity_detection_(VAD).ipynb)
- [Prepare Language Model](https://github.com/huseinzol05/malaya-speech/blob/b44d08a225ce9ea6881527cd66018453feb1ace4/pretrained-model/stt/prepare-lm/README.md#L10)

4. Issue

    4.1. The model not able to recognize my name/company/brand
    - The reason why the model not able to recognize because it is not in the training dataset, you can create kenlm language model to make the model recognize it correctly or use Hotword with custom weight to correctly recognize it. See tutorials/2. speech to text with language model.ipynb

    4.2. The model not able to recognize common word.
    - The reason might be the word not in the training set, you can make the model predict correctly by following above suggestion or create an issue with the audio and text(or text only) so that we can make it work and add as our evaluation dataset.

    4.3. Need feature X
    - Can create issue with example application and we will consider to add it in the next version.

    4.4. How to improve the model prediction?
    - You can create an issue and share with us reproducible step to that lead to wrong prediction so that we can debug the issue or you can create your own language model to improve the model prediction. Currently we provide common word language model if you use "sk filepath --decoder v1" in cli or "predict(filepath,decoder='v1')" in python code

    4.5. Want to contribute (Data,Compute power,Annotation,Features)
    - Can contact us at khursani@omesti.com

References:

1. ONNX optimization based on https://mp.weixin.qq.com/s/ZLZ4F2E_wYEMODGWzdhDRg
2. https://github.com/NVIDIA/NeMo
3. https://github.com/alphacep/vosk-server/

Related:

1. https://github.com/huseinzol05/Malaya-Speech
2. https://github.com/bagustris/id
