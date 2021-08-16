# suara-kami-community

Suara Kami: pre-trained STT models for Bahasa Malaysia.

A simple pipeline for doing speech processing
Works for GPU and CPU
Small and fast
Without pytorch or tf dependencies
support multiple audio type

1. Setup

```
pip install https://github.com/redapesolutions/suara-kami-community
or
git clone https://github.com/redapesolutions/suara-kami-community
cd suara-kami-community
pip install . --upgrade
```

2. Usage

```
Usage: sk FN <flags>
  optional flags:        --model | --decoder | --output_folder | --output_csv |
                         --audio_type

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

GPU Usage

```
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

GRPC Server/Client
check server/grpc folder

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

3. Issue

    3.1. The model not able to recognize my name/company/brand
    - The reason why the model not able to recognize because it is not in the training dataset, you can create kenlm language model to make the model recognize it correctly or use Hotword with custom weight to correctly recognize it. See tutorials/2. speech to text with language model.ipynb

    3.2. The model not able to recognize common word.
    - The reason might be the word not in the training set, you can make the model predict correctly by following above suggestion or create an issue with the audio and text(or text only) so that we can make it work and add as our evaluation dataset.

    3.3. Need feature X
    - Can create issue and we will consider to add it in the next version.

    3.4. Want to contribute (Data,Compute power,Annotation,Features)
    - Can contact us at khursani@omesti.com
     
References:

1. ONNX optimization based on https://mp.weixin.qq.com/s/ZLZ4F2E_wYEMODGWzdhDRg
2. https://github.com/NVIDIA/NeMo
3. https://github.com/alphacep/vosk-server/

Related:

1. https://github.com/huseinzol05/Malaya-Speech
2. https://github.com/bagustris/id