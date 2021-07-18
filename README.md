# suara-kami-community

Suara Kami: pre-trained STT models for Bahasa Malaysia.

A simple pipeline for doing speech processing
Works for GPU and CPU
Small and fast

TODO
Tutorial for:
[/] Speech To Text
[] Text To Speech
[] Speech Enhancement -> Remove noise
[] Speaker Conversion -> Change speaker voice 
[] Speech Translation -> Change speech from source lang to target lang
[] Sentiment Analysis -> Extraction sentiment based on the speech
[] Topic Modeling     -> Extract topic from discussion
[] Voice Activity Detection -> Detect speech activity at certain frame

1. Setup

1.1 Download model

CER and WER calculated using Jiwer

| name               | loss  | cer | wer  | entropy |  Size | Link | Summary                                                                  |
| ------------------ | ----- | --- | ---- | ------- |  ---- | ---- | ------------------------------------------------------------------------ |
| Conformer tiny     | ctc   | 11  | 40   | 0.8     |  10MB |      | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small    | w-ctc | 6.1 | 23.9 | 0.6     |  50MB |      | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small-lm | w-ctc | 0.3 | 14   | -       |  50MB |      | 457422 of audio files with total duration of 620hours 6minutes 51seconds |
| Conformer small    | rnnt  | TODO| TODO | TODO    |       |      |                                                                          |

* All model trained on Google Colab with limited number of dataset because of Google Colab storage space limitation

1. Inference

Inference to a folder
'''
run code that point to a folder
python3 predict.py --path /content/test
'''

Inference to one file
'''
run code that point to a wav file
python3 predict.py --path /content/test/youtube/0.wav
'''

2. Issue

1. The model not able to recognize my name/company/brand
- The reason why the model not able to recognize because it is not in the training dataset, you can create kenlm language model to make the model recognize it correctly or use Hotword with custom weight to correctly recognize it.

2. The model not able to recognize common word.
- The reason might be the word not in the training set, you can make the model predict correctly by following above suggestion or create an issue with the audio and text(or text only) so that we can make it work and add as our evaluation dataset.

3. 