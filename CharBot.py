import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as f:
    data = json.load(f)


# try except so we dont have to manage data everytime
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for question in intent["patterns"]:
            word = nltk.word_tokenize(question)   # take each question and bring it to root word
            words.extend(word)                    # this record each word in the patterns
            docs_x.append(word)                   # this is a 2D array that break each question into words and save in an array
            docs_y.append(intent["tag"])          # classify question vector for later

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] # conver every word into lower case
    words = sorted(list(set(words)))                 # set remove duplicate

    labels = sorted(labels)

    # input is a list that show if a word exist
    # output 

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # bag of words if a word appear put 1
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        # create a vector for each question, index 1 if word in question appeared in the words
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]                    # make a copy
        output_row[labels.index(docs_y[x])] = 1      # classify the tag of output

        training.append(bag)          # all the training question
        output.append(output_row)     # the tage vector associeated with each training eqestion

    # our model needs array
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# input layer of the neuron input is the length of total number of words
net = tflearn.input_data(shape=[None, len(training[0])])
# first hidden layer
net = tflearn.fully_connected(net,8)
# second hidden layer
net = tflearn.fully_connected(net,8)
# output layer
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")  # sofemax gives probability of each neuron and we choose the highest prob
net = tflearn.regression(net)

model = tflearn.DNN(net)

# loading the bot
model.load("ChatBot.tflearn")

# uncomment to train the bot
#model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
#model.save("ChatBot.tflearn")

# turn input sentense into bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    question = nltk.word_tokenize(s)
    question = [stemmer.stem(word.lower()) for word in question]

    for word in question:
        for i,w in enumerate(words):
            if w == word:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("You are connected to my personal assistance! (type \"quit\" to exit)")
    while True:
        inp = input("Please ask any question: ")
        if inp.lower() == "quit":
            break

        prediction = model.predict([bag_of_words(inp, words)])[0]

        prediction_ind = np.argmax(prediction)
        if prediction[prediction_ind] > 0.7:
            tag = labels[prediction_ind]
                
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                
            print(random.choice(responses))
        else:
            print("Sorry, I don't understand. Please ask me another question.")

chat()
