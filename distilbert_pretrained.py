#https://analyticsindiamag.com/python-guide-to-huggingface-distilbert-smaller-faster-cheaper-distilled-bert/
#https://huggingface.co/docs/transformers/model_doc/distilbert

from transformers import DistilBertTokenizer, DistilBertModel, \
DistilBertForMaskedLM, DistilBertForSequenceClassification, \
DistilBertForQuestionAnswering, DistilBertForMultipleChoice, \
DistilBertForTokenClassification
from jtop import jtop, JtopException
import torch
import numpy as np

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
  tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
  model = DistilBertModel.from_pretrained("distilbert-base-uncased")
  inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

  #GPU-WARM-UP
  for _ in range(10):
    _ = model(**inputs)

  for rep in range(repetitions):
    # Measure start time
    starter.record()
    model(**inputs)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
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

  model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
  inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
  labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

  #GPU-WARM-UP
  for _ in range(10):
    _ = model(**inputs, labels=labels)

  for rep in range(repetitions):
    # Measure start time
    starter.record()
    inference = model(**inputs, labels=labels)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
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

  avg_inference_time = sum(timings) / repetitions

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

  model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
  inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
  labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

  #GPU-WARM-UP
  for _ in range(10):
    _ = model(**inputs, labels=labels)

  for rep in range(repetitions):
    # Measure start time
    starter.record()
    model(**inputs, labels=labels)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    timings[rep] = starter.elapsed_time(ender)

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

  avg_inference_time = sum(timings) / repetitions

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

  model = DistilBertForMultipleChoice.from_pretrained('distilbert-base-cased')
  prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
  choice0 = "It is eaten with a fork and a knife."
  choice1 = "It is eaten while held in the hand."
  labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
  encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors='pt', padding=True)
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

  for _ in range(10):
      _ = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels) 

  for rep in range(repetitions):
    # Measure start time
    starter.record()
    model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels) 
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
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

  avg_inference_time = sum(timings) / repetitions

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

  model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased')
  inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
  labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

  #GPU-WARM-UP
  for _ in range(10):
      _ = model(**inputs, labels=labels)

  for rep in range(repetitions):
    # Measure start time
    starter.record()
    model(**inputs, labels=labels)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
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

  avg_inference_time = sum(timings) / repetitions

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

  model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
  question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
  inputs = tokenizer(question, text, return_tensors='pt')
  start_positions = torch.tensor([1])
  end_positions = torch.tensor([3])
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

  #GPU-WARM-UP
  for _ in range(10):
      _ = model(**inputs, start_positions=start_positions, end_positions=end_positions)

  for rep in range(repetitions):
    # Measure start time
    starter.record()
    model(**inputs, start_positions=start_positions, end_positions=end_positions)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
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

  avg_inference_time = sum(timings) / repetitions

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
