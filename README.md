# suara-kami-community

Suara Kami: pre-trained STT models for Bahasa Malaysia.

A simple pipeline for doing speech processing
Works for GPU and CPU
Small and fast

1. Setup

```
pip install https://github.com/redapesolutions/suara-kami-community
or
git clone https://github.com/redapesolutions/suara-kami-community
cd suara-kami-community
pip install . --upgrade
```

2. Usage

Inference to a folder
run code that point to a folder containing wav file
```
sk conformer_small audio_folder_path
or
sk conformer_small audio_folder_path --output_folder output
or
sk conformer_small audio_folder_path --output_folder output --output_csv results.csv
```

Inference to one file
run code that point to a wav file
```
sk conformer_small audio_path.wav
```

CER and WER calculated using Jiwer

| name               | loss  | cer | wer  | entropy |  Size | Link | Summary                                                                  |
| ------------------ | ----- | --- | ---- | ------- |  ---- | ---- | ------------------------------------------------------------------------ |
| Conformer tiny     | ctc   | 11  | 40   | 0.8     |  10MB |      | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small    | w-ctc | 6.1 | 23.9 | 0.6     |  50MB |      | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small-lm | w-ctc | 0.3 | 14   | -       |  50MB |      | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small    | rnnt  | TODO| TODO | TODO    |       |      |                                                                          |

* All model trained on Google Colab with limited number of dataset because of Google Colab storage space limitation

3. Issue

    3.1. The model not able to recognize my name/company/brand
    - The reason why the model not able to recognize because it is not in the training dataset, you can create kenlm language model to make the model recognize it correctly or use Hotword with custom weight to correctly recognize it. See tutorials/2. speech to text with language model.ipynb

    3.2. The model not able to recognize common word.
    - The reason might be the word not in the training set, you can make the model predict correctly by following above suggestion or create an issue with the audio and text(or text only) so that we can make it work and add as our evaluation dataset.

    3.3. Need feature X
    - Can create issue and we will consider add it in the next version.

    3.4. Want to contribute (Data,Compute power,Annotation,Features)
    - Can contact us at khursani@omesti.com
     
References:

1. ONNX optimization based on https://mp.weixin.qq.com/s/ZLZ4F2E_wYEMODGWzdhDRg
2. 

Related:

1. https://github.com/huseinzol05/Malaya-Speech
2. https://github.com/bagustris/id