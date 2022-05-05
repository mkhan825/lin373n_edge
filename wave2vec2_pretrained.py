import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# print(model)
#loading the audio file
speech, rate = librosa.load('hero.wav',sr=16000)
# print(speech)
# print(rate)
input_values = tokenizer(speech, return_tensors = 'pt').input_values
print(input_values)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 3
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(input_values)

for rep in range(repetitions):
  # Measure start time
  starter.record()
  inference = model(input_values)
  ender.record()
  # WAIT FOR GPU SYNC
  torch.cuda.synchronize()
  curr_time = starter.elapsed_time(ender)
  timings[rep] = curr_time
#Store logits (non-normalized predictions)
predict = model(input_values) 
print(predict)
logits = predict.logits
print(logits)
#Store predicted id's
predicted_ids = torch.argmax(logits, dim =-1)
#decode the audio to generate text
print(predicted_ids)
transcriptions = tokenizer.decode(predicted_ids[0])
print("Predicted: ",transcriptions)
print("Actual: You either die a hero, or you live long enough to see yourself become the villain")
total_inference_time = 0
for time in timings:
  total_inference_time += time
print(f"Average inference time: {total_inference_time/repetitions}")