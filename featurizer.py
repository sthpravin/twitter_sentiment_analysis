from sklearn.feature_extraction.text import CountVectorizer
import pickle

class Featurize:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def vectorize_train(self, input):
        features = self.vectorizer.fit_transform(input)
        return features

    def vectorize_test(self, input):
        features = self.vectorizer.transform(input)
        return features

