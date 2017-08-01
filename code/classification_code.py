import csv
from keras.models import load_model
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
from keras.models import Model


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

print("reading document_set")
reader = csv.reader(open('document_set.csv', 'r'))
document_set = {}
next(reader)
for row in reader:
   k, v = row
   document_set[k] = v


print("reading test set now")
reader1 = csv.reader(open('Test_Data.csv', 'r'))
test_set = {}
next(reader1)
for row in reader1:
   k, v = row
   test_set[k] = document_set[k]

print("done ...  predicting now")

model = load_model('weights.best.hdf5')

print('Processing text dataset')

texts = []  # list of text samples
texts = list(test_set.values())
docs = list(test_set.keys())

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


print('Shape of data tensor:', data.shape)

preds = model.predict(data)
preds = preds.argmax(axis=-1)
preds = preds.tolist()
print("preds done")



with open('preds.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(docs, preds))