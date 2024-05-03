import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from preprocess import TextPreprocessor
import joblib

class SentimentClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.clf.fit(X_train_vectorized, y_train)

    def evaluate(self, X_test, y_test):
        X_test_vectorized = self.vectorizer.transform(X_test)
        accuracy = self.clf.score(X_test_vectorized, y_test)
        return accuracy
    
    def save_model(self, filepath):
        joblib.dump((self.vectorizer, self.clf), filepath)

    def load_model(self, filepath):
        self.vectorizer, self.clf = joblib.load(filepath)

if __name__ == "__main__":
    data = pd.read_csv('reviews.tsv', delimiter='\t')
    X = data['text']
    y = data['sentiment']

    preprocessor = TextPreprocessor()
    X_processed = X.apply(preprocessor.preprocess)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    classifier = SentimentClassifier()
    classifier.train(X_train, y_train)

    accuracy = classifier.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')

    classifier.save_model('sentiment_model.joblib')
