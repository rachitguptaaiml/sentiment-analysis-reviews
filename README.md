# Product Review Sentiment Analyzer

This project uses TF-IDF and Logistic Regression for sentiment analysis on product reviews.

## Features

* Loads dataset of product reviews with corresponding sentiments (positive or negative)
* Preprocesses text data using TF-IDF vectorization
* Splits data into training and testing sets
* Trains a Logistic Regression model to classify new reviews as positive or negative
* Evaluates model performance using accuracy score
* Provides sample output for classifying new review

## Setup and Run Instructions

1. Clone repository: `git clone https://github.com/your-username/sentiment-analysis-reviews.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Load dataset from reviews.csv file (assuming it's in the same directory as main.py)
4. Run main.py using Python 3.x: `python main.py`

## Sample Output

Model Accuracy: 0.933
Review is positive

## What I Learned

Through this project, I gained hands-on experience with sentiment analysis using TF-IDF and Logistic Regression. I learned the importance of preprocessing text data, selecting appropriate parameters for vectorization, and tuning hyperparameters to achieve optimal model performance. This project helped me develop a solid foundation in machine learning and deepened my understanding of natural language processing techniques.