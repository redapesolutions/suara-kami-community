## based on code from https://github.com/snakers4/silero-vad

from collections import deque
import numpy as np

def validate_onnx(model, inputs):
    ort_inputs = {'input': inputs}
    outs = model.run(None, ort_inputs)
    return outs[0]

def collect_chunks(tss,wav):
    chunks = []
    for i in tss:
        chunks.append(wav[i['start']: i['end']])
    return np.concatenate(chunks)

def get_speech_ts_adaptive(wav,
                      model,
                      batch_size: int = 200,
                      step: int = 500,
                      num_samples_per_window: int = 4000, # Number of samples per audio chunk to feed to NN (4000 for 16k SR, 2000 for 8k SR is optimal)
                      min_speech_samples: int = 10000,  # samples
                      min_silence_samples: int = 4000,
                      speech_pad_samples: int = 2000,
                      run_function = validate_onnx,
                      visualize_probs=False,
    ):
    if visualize_probs:
      import pandas as pd

    num_samples = num_samples_per_window
    num_steps = int(num_samples / step)
    assert min_silence_samples >= step
    outs = []
    to_concat = []
    for i in range(0, len(wav), step):
        chunk = wav[i: i+num_samples]
        if len(chunk) < num_samples:
            chunk = np.pad(chunk, (0, num_samples - len(chunk)))
        to_concat.append(chunk[None])
        if len(to_concat) >= batch_size:
            chunks = np.concatenate(to_concat,axis=0)
            out = run_function(model, chunks)
            outs.append(out)
            to_concat = []
    if to_concat:
        chunks = np.concatenate(to_concat,axis=0)
        out = run_function(model, chunks)
        outs.append(out)

    outs = np.concatenate(outs, axis=0)

    buffer = deque(maxlen=num_steps)
    triggered = False
    speeches = []
    smoothed_probs = []
    current_speech = {}
    speech_probs = outs[:, 1]  # 0 index for silence probs, 1 index for speech probs
    median_probs = np.median(speech_probs)

    trig_sum = 0.89 * median_probs + 0.08 # 0.08 when median is zero, 0.97 when median is 1

    temp_end = 0
    for i, predict in enumerate(speech_probs):
        buffer.append(predict)
        smoothed_prob = max(buffer)
        if visualize_probs:
            smoothed_probs.append(float(smoothed_prob))
        if (smoothed_prob >= trig_sum) and temp_end:
            temp_end = 0
        if (smoothed_prob >= trig_sum) and not triggered:
            triggered = True
            current_speech['start'] = step * max(0, i-num_steps)
            continue
        if (smoothed_prob < trig_sum) and triggered:
            if not temp_end:
                temp_end = step * i
            if step * i - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue
    if current_speech:
        current_speech['end'] = len(wav)
        speeches.append(current_speech)
    if visualize_probs:
        pd.DataFrame({'probs': smoothed_probs}).plot(figsize=(16, 8))

    for i, ts in enumerate(speeches):
        if i == 0:
            ts['start'] = max(0, ts['start'] - speech_pad_samples)
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - ts['end']
            if silence_duration < 2 * speech_pad_samples:
                ts['end'] += silence_duration // 2
                speeches[i+1]['start'] = max(0, speeches[i+1]['start'] - silence_duration // 2)
            else:
                ts['end'] += speech_pad_samples
        else:
            ts['end'] = min(len(wav), ts['end'] + speech_pad_samples)

    return speeches
