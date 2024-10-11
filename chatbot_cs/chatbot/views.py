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

# Load the intents and model
with open("intents.json") as file:
    data = json.load(file)

model = keras.models.load_model("chatbot_model.h5")

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chatbot_response(request):
    if request.method == "POST":
        user_message = request.POST.get('message')
        results = model.predict(np.array([bag_of_words(user_message, words)]))
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

