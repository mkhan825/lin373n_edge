import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
#loading the audio file
speech, rate = librosa.load('/content/drive/My Drive/LIN373N/hero.wav',sr=16000)
input_values = tokenizer(speech, return_tensors = 'pt').input_values
#Store logits (non-normalized predictions)
logits = model(input_values).logits
#Store predicted id's
predicted_ids = torch.argmax(logits, dim =-1)
#decode the audio to generate text
transcriptions = tokenizer.decode(predicted_ids[0])
print("Predicted: ",transcriptions)
print("Actual: You either die a hero, or you live long enough to see yourself become the villain")