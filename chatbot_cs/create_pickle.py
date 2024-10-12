import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import json
import pickle

stemmer = LancasterStemmer()

# Load intents from the JSON file
with open("intents.json") as file:
    intents = json.load(file)

words = []
labels = []
docs_x = []  # For storing tokenized patterns
docs_y = []  # For storing corresponding tags

# Process the intents and fill words, docs_x, docs_y
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)  # Add to words list
        docs_x.append(wrds)  # Save tokenized pattern
        docs_y.append(intent['tag'])  # Save corresponding tag
    
    # Add each tag to the labels list (if it's not already present)
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Stem and lower each word, and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

# Sort labels
labels = sorted(labels)

# Create training data
training = []
output = []

# Create an empty output array for the correct label (one-hot encoded)
out_empty = [0 for _ in range(len(labels))]

# Create bag of words for each pattern
for idx, doc in enumerate(docs_x):
    bag = []

    # Stem each word in the pattern
    wrds = [stemmer.stem(w.lower()) for w in doc]

    # Create the bag of words array
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    # Create the one-hot output array
    output_row = out_empty[:]
    output_row[labels.index(docs_y[idx])] = 1

    training.append(bag)
    output.append(output_row)

# Convert training and output to numpy arrays
training = np.array(training)
output = np.array(output)

# Save the words, labels, training, and output data into a pickle file
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

print("Correct data.pickle created successfully.")

