from utils import Preprocessor
from utils import save_model
from utils import loader
from featurizer import Featurize
from sentiment_analyzer import NaiveBayes
from sklearn.model_selection import train_test_split
from evaluator import evaluate_accuracy
import pickle


X, y = loader()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)

token = Preprocessor()
file1 = open('preprocessor.obj','wb')
pickle.dump(token, file1)
file1.close()

train_preprocessed = [token.tweet_cleaner(i) for i in X_train]
f = Featurize()
train_features = f.vectorize_train(train_preprocessed)
file2 = open('featurizer.obj', 'wb')
pickle.dump(f, file2)
file2.close()

model = NaiveBayes()
clf = model.train(train_features, labels = y_train)
save_model(clf)
print('Model is trained')

test_preprocessed= [token.tweet_cleaner(i) for i in X_test]
test_features = f.vectorize_test(test_preprocessed)
y_predicted = model.predict(test_features)
print("test accuracy : " + str(evaluate_accuracy(y_test, y_predicted)))