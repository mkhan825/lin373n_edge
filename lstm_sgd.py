#LSTM with stochastic gradient descent
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import os
from keras.models import Sequential
from keras import layers
import time

filepath_dict = {'yelp':   'sentiment-labelled-sentences/yelp_labelled.txt',
                 'amazon': 'sentiment-labelled-sentences/amazon_cells_labelled.txt',
                 'imdb':   'sentiment-labelled-sentences/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
  df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
  df['source'] = source  # Add another column filled with the source name
  df_list.append(df)

df = pd.concat(df_list)
print(df.head())

sentences = df['sentence'].values
y = df['label'].values
# First split 20% of the data into testing and validation
sentences_train, sentences_test_val, y_train, y_test_val = train_test_split(sentences, y, test_size=0.3, random_state=1000)
# Then split 50% of the test+val data as validation
sentences_test, sentences_val, y_test, y_val = train_test_split(sentences_test_val, y_test_val, test_size=0.5, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_val = vectorizer.transform(sentences_val)
input_dim = X_train.shape[1]

# Training
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
start_time = time.time()
model.fit(X_train, y_train, 
          epochs=10, 
          validation_data=(X_val, y_val), ## specifying the validation set 
          batch_size=20)
end_time = time.time()
loss, accuracy = model.evaluate(X_test, y_test)
print(f"It took {end_time - start_time} ms to train lstm with adam with {accuracy} accuracy")
