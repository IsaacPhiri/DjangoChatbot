import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import json
import pickle

# Initialize stemmer
stemmer = LancasterStemmer()

# Ensure necessary NLTK data is downloaded
nltk.download('punkt_tab')

# Load the intents file
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Define the model architecture using Keras
model = keras.Sequential()
model.add(keras.layers.Input(shape=(len(training[0]),)))  # Input layer
model.add(keras.layers.Dense(8, activation='relu'))       # First hidden layer
model.add(keras.layers.Dense(8, activation='relu'))       # Second hidden layer
model.add(keras.layers.Dense(len(output[0]), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(training, output, epochs=1000, batch_size=8)

# Save the model
model.save("chatbot_model.h5")
