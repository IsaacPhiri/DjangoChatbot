from django.shortcuts import render
from django.http import JsonResponse
import random
import json
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow import keras
import pickle

stemmer = LancasterStemmer()

nltk.download('punkt') 
nltk.download('punkt_tab')

# Load the intents and model
with open("intents.json") as file:
    data = json.load(file)

model = keras.models.load_model("chatbot_model.keras")

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# def bag_of_words(s, words):
#     bag = [0 for _ in range(len(words))]
#     s_words = nltk.word_tokenize(s)
#     s_words = [stemmer.stem(word.lower()) for word in s_words]

#     for se in s_words:
#         for i, w in enumerate(words):
#             if w == se:
#                 bag[i] = 1
#     return np.array(bag)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]  # Initialize a bag of zeroes

    s_words = nltk.word_tokenize(s)  # Tokenize input sentence
    s_words = [stemmer.stem(word.lower()) for word in s_words]  # Stem the words

    # Populate the bag with 1s where the stemmed word matches 'words'
    for se in s_words:
        if se in words:
            index = words.index(se)
            bag[index] = 1

    # Debugging: check the length and contents of the bag
    print("Input sentence:", s)
    print("Bag of words vector:", bag)
    print("Length of bag of words vector:", len(bag))

    return np.array(bag)

def chatbot_response(request):
    if request.method == "POST":
        user_message = request.POST.get('message')
        
        # Debugging: check bag of words output before passing to model
        bow_vector = bag_of_words(user_message, words)
        print("Bag of words vector shape:", bow_vector.shape)  # Should be (58,)

        # Reshape the input to match the model's expected input shape
        results = model.predict(np.array([bow_vector]))  # Reshape to (1, 58)
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        response = random.choice(responses)
        return JsonResponse({"response": response})
    return JsonResponse({"response": "Invalid request method"})


# Create a view for rendering the chatbot frontend
def index(request):
    return render(request, 'chatbot/index.html')

