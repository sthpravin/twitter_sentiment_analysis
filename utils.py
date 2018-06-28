import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re

class Preprocessor:
    def __init__(self):
        self.tok = WordPunctTokenizer()
        self.pat1 = r'@[A-Za-z0-9]+'
        self.pat2 = r'https?://[A-Za-z0-9./]+'
        self.combined_pat = r'|'.join((self.pat1, self.pat2))

    def tweet_cleaner(self, text):
        soup = BeautifulSoup(text,'lxml')
        souped = soup.get_text()
        stripped = re.sub(self.combined_pat, ' ', souped)
        try:
            clean = stripped.decode("utf-8-sig").replace(u"\ufffd","?")
        except:
            clean = stripped
        letters_only = re.sub("[^a-zA-Z]", " ", clean)
        lower_case = letters_only.lower()
        words = self.tok.tokenize(lower_case)
        return (" ".join(words)).strip()


def loader():
    dataset = pd.read_csv('data/train.csv', encoding = 'latin-1')
    dataset.drop(['ItemID'], axis=1, inplace=True)
    X = dataset['SentimentText']
    y = dataset['Sentiment']
    return X,y


def save_model(clf):
    joblib.dump(clf, 'models/model.pkl')

def load_model():
    return(joblib.load('models/model.pkl'))