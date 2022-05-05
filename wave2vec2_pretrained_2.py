import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from jtop import jtop, JtopException
import numpy as np

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
speech, rate = librosa.load('burn.wav',sr=16000)
input_values = tokenizer(speech, return_tensors = 'pt').input_values
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 100

timings=np.zeros((repetitions,1))

temps_cpu = np.zeros((repetitions,1))
temps_gpu = np.zeros((repetitions,1))

freq_cpu = np.zeros((repetitions,1))
freq_gpu = np.zeros((repetitions,1))

power_cur = np.zeros((repetitions,1))
power_avg = np.zeros((repetitions,1))

power_cur_gpu = np.zeros((repetitions,1))
power_avg_gpu = np.zeros((repetitions,1))

power_cur_cpu = np.zeros((repetitions,1))
power_avg_cpu = np.zeros((repetitions,1))

with jtop() as jetson:
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
    timings[rep] = curr_time

    temps_cpu[rep] = jetson.stats['Temp CPU']
    temps_gpu[rep] = jetson.stats['Temp GPU']

    freq_cpu[rep] = (jetson.cpu['CPU1']['frq'] + jetson.cpu['CPU2']['frq'] + jetson.cpu['CPU3']['frq'] + jetson.cpu['CPU4']['frq']) / 4
    freq_gpu[rep] = jetson.gpu['frq']

    power_cur[rep] = jetson.power[0]['cur']
    power_avg[rep] = jetson.power[0]['avg']

    power_cur_gpu[rep] = jetson.power[1]['5V GPU']['cur']
    power_avg_gpu[rep] = jetson.power[1]['5V GPU']['avg']
    power_cur_cpu[rep] = jetson.power[1]['5V CPU']['cur']
    power_avg_cpu[rep] = jetson.power[1]['5V CPU']['avg']

  avg_inference_time = sum(timings)[0] / repetitions

  avg_temps_cpu = sum(temps_cpu)[0]/repetitions
  avg_temps_gpu = sum(temps_gpu)[0]/repetitions

  avg_freq_cpu = sum(freq_cpu)[0]/repetitions
  avg_freq_gpu = sum(freq_gpu)[0]/repetitions

  avg_power_cur = sum(power_cur)[0]/repetitions
  avg_power_avg = sum(power_avg)[0]/repetitions

  avg_power_cur_gpu = sum(power_cur_gpu)[0]/repetitions
  avg_power_avg_gpu = sum(power_avg_gpu)[0]/repetitions
  avg_power_cur_cpu = sum(power_cur_cpu)[0]/repetitions
  avg_power_avg_cpu = sum(power_avg_cpu)[0]/repetitions

  print("")
  print(f"Average inference time for tokenizer: {avg_inference_time}")
  print(f"Average temperature for CPU: {avg_temps_cpu}")
  print(f"Average temperature for GPU: {avg_temps_gpu}")
  print(f"Average frequency for CPU: {avg_freq_cpu}")
  print(f"Average frequency for CPU: {avg_freq_gpu}")
  print(f"Average current power: {avg_power_cur}")
  print(f"Average power averaged out over time: {avg_power_avg}")
  print(f"Average current power for GPU: {avg_power_cur_gpu}")
  print(f"Average GPU power averaged out over time: {avg_power_avg_gpu}")
  print(f"Average current power for CPU: {avg_power_cur_cpu}")
  print(f"Average CPU power averaged out over time: {avg_power_avg_cpu}")

  #Store logits (non-normalized predictions)
  logits = inference.logits
  #Store predicted id's
  predicted_ids = torch.argmax(logits, dim =-1)
  #decode the audio to generate text
  transcriptions = tokenizer.decode(predicted_ids[0])
  transcription_text = transcriptions.lower()
  real_text = "some men aren't looking for anything logical like money they can't be bought bullied reasoned or negotiated with some men just want to watch the world burn"

  print("Predicted: ",transcription_text)
  print("Actual: ", real_text)
