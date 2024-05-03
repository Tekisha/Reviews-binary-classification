# Sentiment Analysis with Naive Bayes Classifier

This project demonstrates sentiment analysis using a Naive Bayes classifier trained on movie and book reviews. The classifier predicts whether a review is positive or negative based on its text content.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/Tekisha/Reviews-binary-classification.git
    ```

2. Navigate to the project directory:
    ```
    cd Reviews-binary-classification
    ```


3. Create a virtual environment:
    ```
    python -m venv venv
    ```


4. Activate the virtual environment:
- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source venv/bin/activate
  ```

5. Install dependencies:
    ```
    pip install -r requirements.txt
    ```


## Usage

1. Run the script to train the classifier and evaluate its accuracy:
    ```
    python train_model.py
    ```

2. Once the classifier is trained, you can use it to predict sentiment for new reviews:
    ```
    python predict.py
    ```

## Model Accuracy
The accuracy of the trained classifier on the test data is approximately 96.58%.

## Predicting Sentiment for New Reviews
After training the classifier, you can use it to predict the sentiment of new reviews by providing the text of the review. Update the **new_reviews** list in the `predict.py` script with the new reviews you want to predict sentiment for, and run the script.

## Files
- `preprocess.py`: Preprocessing functions for tokenization and stemming.
- `train_model.py`: Class definition for the Naive Bayes classifier.
- `predict.py`: Script for predicting sentiment for new reviews.
- `requirements.txt`: List of required dependencies.
- `README.md`: This README file.
- `.gitignore`: File to specify which files and directories to ignore in version control.
- `sentiment_model.joblib`: Serialized trained classifier model.

## Author
Teodor Vidaković - [GitHub](https://github.com/Tekisha)