#https://analyticsindiamag.com/python-guide-to-huggingface-distilbert-smaller-faster-cheaper-distilled-bert/
#https://huggingface.co/docs/transformers/model_doc/distilbert

from transformers import DistilBertTokenizer, DistilBertModel, \
DistilBertForMaskedLM, DistilBertForSequenceClassification, \
DistilBertForQuestionAnswering, DistilBertForMultipleChoice, \
DistilBertForTokenClassification
import torch
import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(**inputs)
inference = None
for rep in range(repetitions):
  # Measure start time
  starter.record()
  inference = model(**inputs)
  ender.record()
  # WAIT FOR GPU SYNC
  torch.cuda.synchronize()
  curr_time = starter.elapsed_time(ender)
  timings[rep] = curr_time
outputs = inference 
# print(outputs.loss)
last_hidden_states = outputs.last_hidden_state
total_inference_time = 0
for time in timings:
  total_inference_time += time
print(f"Average inference time for tokenizer: {total_inference_time/repetitions}")

model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
#INITS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(**inputs, labels=labels)
inference = None
for rep in range(repetitions):
  # Measure start time
  starter.record()
  inference = model(**inputs, labels=labels)
  ender.record()
  # WAIT FOR GPU SYNC
  torch.cuda.synchronize()
  curr_time = starter.elapsed_time(ender)
  timings[rep] = curr_time
outputs = inference 
print(outputs.loss)
loss = outputs.loss
logits = outputs.logits 
total_inference_time = 0
for time in timings:
  total_inference_time += time
print(f"Average inference time for MaskedLM: {total_inference_time/repetitions}")


model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#INITS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(**inputs, labels=labels)
inference = None
for rep in range(repetitions):
  # Measure start time
  starter.record()
  inference = model(**inputs, labels=labels)
  ender.record()
  # WAIT FOR GPU SYNC
  torch.cuda.synchronize()
  curr_time = starter.elapsed_time(ender)
  timings[rep] = curr_time
outputs = inference 
print(outputs.loss)
loss = outputs.loss
logits = outputs.logits 
total_inference_time = 0
for time in timings:
  total_inference_time += time
print(f"Average inference time for SequenceClassification: {total_inference_time/repetitions}")

model = DistilBertForMultipleChoice.from_pretrained('distilbert-base-cased')
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors='pt', padding=True)
#INITS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels) 
inference = None
for rep in range(repetitions):
  # Measure start time
  starter.record()
  inference = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels) 
  ender.record()
  # WAIT FOR GPU SYNC
  torch.cuda.synchronize()
  curr_time = starter.elapsed_time(ender)
  timings[rep] = curr_time
outputs = inference # batch size is 1
print(outputs.loss)
# the linear classifier still needs to be trained
loss = outputs.loss
logits = outputs.logits 
total_inference_time = 0
for time in timings:
  total_inference_time += time
print(f"Average inference time for MultipleChoice: {total_inference_time/repetitions}")


model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
#INITS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(**inputs, labels=labels)
inference = None
for rep in range(repetitions):
  # Measure start time
  starter.record()
  inference = model(**inputs, labels=labels)
  ender.record()
  # WAIT FOR GPU SYNC
  torch.cuda.synchronize()
  curr_time = starter.elapsed_time(ender)
  timings[rep] = curr_time
outputs = inference
print(outputs.loss)
loss = outputs.loss
logits = outputs.logits 
total_inference_time = 0
for time in timings:
  total_inference_time += time
print(f"Average inference time for TokenClassification: {total_inference_time/repetitions}")

model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors='pt')
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
#INITS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(**inputs, start_positions=start_positions, end_positions=end_positions)
inference = None
for rep in range(repetitions):
  # Measure start time
  starter.record()
  inference = model(**inputs, start_positions=start_positions, end_positions=end_positions)
  ender.record()
  # WAIT FOR GPU SYNC
  torch.cuda.synchronize()
  curr_time = starter.elapsed_time(ender)
  timings[rep] = curr_time
outputs = inference 
print(outputs.loss)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
total_inference_time = 0
for time in timings:
  total_inference_time += time
print(f"Average inference time for QuestionAnswering: {total_inference_time/repetitions}")
# print(start_scores)
# print(end_scores)
