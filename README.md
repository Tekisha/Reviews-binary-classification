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