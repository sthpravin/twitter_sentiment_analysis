from utils import Preprocessor
from utils import load_model
from featurizer import Featurize
from sentiment_analyzer import NaiveBayes
import pickle
import sys
import numpy as np
from nltk import sent_tokenize

def evaluate(value):



    test = sent_tokenize(value)
    print(test)
    file1 = open('preprocessor.obj', 'rb')
    token = pickle.load(file1)
    file1.close()
    test_preprocessed = [token.tweet_cleaner(i) for i in test]
    print(test_preprocessed)

    file2 = open('featurizer.obj', 'rb')
    f = pickle.load(file2)
    file2.close()
    test_features = f.vectorize_test(test_preprocessed)
    print(test_features)

    model = load_model()
    y_predicted = model.predict(test_features)

#    print(list(y_predicted))
    return y_predicted



def evaluate_accuracy(test_y, predicted_y):
    total_positive = (test_y == predicted_y).sum()
    total = len(test_y)
    accuracy = total_positive/total
    return accuracy

if __name__=="__main__":
    evaluate()