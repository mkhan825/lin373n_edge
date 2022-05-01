#https://analyticsindiamag.com/python-guide-to-huggingface-distilbert-smaller-faster-cheaper-distilled-bert/
#https://huggingface.co/docs/transformers/model_doc/distilbert

from transformers import DistilBertTokenizer, DistilBertModel, \
DistilBertForMaskedLM, DistilBertForSequenceClassification, \
DistilBertForQuestionAnswering, DistilBertForMultipleChoice, \
DistilBertForTokenClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
# print(outputs.loss)
last_hidden_states = outputs.last_hidden_state

model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
outputs = model(**inputs, labels=labels)
print(outputs.loss)
loss = outputs.loss
logits = outputs.logits 

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
print(outputs.loss)
loss = outputs.loss
logits = outputs.logits 

model = DistilBertForMultipleChoice.from_pretrained('distilbert-base-cased')
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors='pt', padding=True)
outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels) # batch size is 1
print(outputs.loss)
# the linear classifier still needs to be trained
loss = outputs.loss
logits = outputs.logits 

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
print(outputs.loss)
loss = outputs.loss
logits = outputs.logits 

model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors='pt')
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
print(outputs.loss)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
# print(start_scores)
# print(end_scores)
