# suara-kami-community

Suara Kami: pre-trained STT models for Bahasa Malaysia.

A simple pipeline for doing speech processing

Works for GPU and CPU

Small and fast

Without pytorch or tf dependencies

support multiple audio type

Can run from CLI or python import

1. Setup

```
pip install https://github.com/redapesolutions/suara-kami-community
or
git clone https://github.com/redapesolutions/suara-kami-community
cd suara-kami-community
pip install . --upgrade
```

2. Usage

Using Python
```
from sk import predict
predict(filepath)
or
predict(filepath,"conformer_tiny")
or 
predict(filepath,decoder="v1")
```

Using Cli
```
Usage: sk FN <flags>
  optional flags:        --model | --decoder | --output_folder | --output_csv |
                         --audio_type

FN              -> filepath
--model         -> model name or model path
--decoder       -> decoder name or decoder path
--output_folder -> transcribed text target location
--output_csv    -> transcribed text csv location
--audio_type    -> if FN is folder, what type of audio format need to search

For detailed information on this command, run:
  sk --help
```

### Inference to one folder or multiple folders
run code that point to a folder containing wav file
```
sk audio_folder_path
or
sk audio_folder_path --output_folder output
or
sk audio_folder_path --output_folder output --output_csv results.csv
or 
sk audio_folder_path,audio_folder_path2
or
sk audio_folder --model conformer_tiny
or
sk audio_folder --audio_type .wav
or
sk audio_folder --audio_type .wav,.mp3 # for supporting other file type
```

### Inference to one or multiple files
run code that point to a wav file
```
sk audio_path.wav
or
sk audio_path.wav,audio_path2.wav
or similar parameter to folder inference
```

### Inference to file and folder
```
sk audio_path.wav,audio_folder
or similar to folder inference
```

### Example
```
➜  suara-kami-community git:(main) ✗ sk /content/test-bahasa/wattpad-audio-wattpad-105.wav           
Can add '--decoder v1' to improve accuracy or prepare your own language model based on README
Total input path: 1
Total audio found(.wav): 1
texts:     ["kejadian ini bukanlah kejadian yang ni"]
filenames: ['/content/test-bahasa/wattpad-audio-wattpad-105.wav']
entropy:   [0.09318003]
timesteps: [[0]]
```
```
➜  suara-kami-community git:(main) ✗ sk /content/test-bahasa/wattpad-audio-wattpad-105.wav --decoder v1
Total input path: 1
Total audio found(.wav): 1
texts:     ["kejadian ini bukanlah kejadian yang ni"]
filenames: ['/content/test-bahasa/wattpad-audio-wattpad-105.wav']
entropy:   [[1.0360475778579712, 0.003976397216320038, 1.0852084159851074, 1.1668410301208496, 0.005958860740065575, 1.022503137588501]]
timestamps: [[["kejadian", 0.01, 0.04], ["ini", 0.04, 0.05], ["bukanlah", 0.06, 1.02], ["kejadian", 1.02, 1.05], ["yang", 1.05, 2.0], ["ni", 2.01, 2.02]]]
```
```
➜  suara-kami-community git:(main) ✗ sk /content/test-bahasa/wattpad-audio-wattpad-105.wav --decoder /content/out.trie.klm 
Total input path: 1
Total audio found(.wav): 1
texts:     ["kejadian ini bukanlah kejadian yang ni"]
filenames: ["/content/test-bahasa/wattpad-audio-wattpad-105.wav"]
entropy:   [[1.0360475778579712, 0.003976397216320038, 1.0852084159851074, 1.1668410301208496, 0.005958860740065575, 1.022503137588501]]
timestamps: [[["kejadian", 0.01, 0.04], ["ini", 0.04, 0.05], ["bukanlah", 0.06, 1.02], ["kejadian", 1.02, 1.05], ["yang", 1.05, 2.0], ["ni", 2.01, 2.02]]]
```

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
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

GRPC Server/Client

check [server/grpc](./server/grpc) folder

Web Example

check [server/web](./server/web) folder

CER and WER calculated using Jiwer

| name               | loss  | cer | wer  | entropy |  Size  | Summary                                                                  |
| ------------------ | ----- | --- | ---- | ------- |  ----  | ------------------------------------------------------------------------ |
| Conformer tiny     | ctc   | 11  | 40   | 0.5     |  18MB  | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer tiny-lm     | ctc   | TODO  | TODO   | TODO     |  18MB  | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small    | w-ctc | 6.1 | 23.9 | 0.6     |  60MB  | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small-lm | w-ctc | 0.3 | 14   | -       |  60MB  | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small    | rnnt  | TODO| TODO | TODO    |        |                                                                          |

* All model trained on Google Colab with limited number of dataset and model size because of Google Colab hardware limitations

Conformer_small: https://zenodo.org/record/5115792 - onnx + tf saved model

3. Tutorials

- [Simple Speech to Text](./tutorials/1.speech_to_text.ipynb)
- [Speech to Text w/ Language model](./tutorials/2.speech_to_text_with_language_model.ipynb)
- [Convert Speech text to Written text form](./tutorials/3.normalize_text.ipynb)
- [Simple Speech to Text using tf saved model](./tutorials/4.tensorflow_inference.ipynb)
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