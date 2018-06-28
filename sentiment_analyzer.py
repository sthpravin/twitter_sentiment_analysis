from sklearn.naive_bayes import MultinomialNB

class NaiveBayes:
    def __init__(self):
        self.clf = MultinomialNB()

    def train(self, features, labels):
        self.clf = self.clf.fit(features, labels)
        return self.clf

    def predict(self, features):
        predicted = self.clf.predict(features)
        return predicted

