import pandas as pd
from train_model import SentimentClassifier
from preprocess import TextPreprocessor

if __name__ == "__main__":
    new_reviews = [
        "This movie was fantastic, I absolutely loved it! movie",
        "The book was really boring, I couldn't finish it.",
        "I'm not sure how I feel about this film, it had its moments but overall it was just okay.",
        "Terrible movie, waste of time and money.",
        "The book was amazing, I couldn't put it down until I finished it."
    ]

    preprocessor = TextPreprocessor()
    new_reviews_processed = [preprocessor.preprocess(review) for review in new_reviews]

    classifier = SentimentClassifier()
    classifier.load_model('sentiment_model.joblib')

    predictions = classifier.clf.predict(classifier.vectorizer.transform(new_reviews_processed))

    for review, prediction in zip(new_reviews, predictions):
        print(f"Review: {review}")
        print(f"Predicted Sentiment: {prediction}")
        print()
