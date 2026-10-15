import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('reviews.csv')

# Preprocess text data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])

y = df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.3f}')

# Use model to classify new review as positive or negative
new_review = 'I love this product! It\'s so easy to use.'
new_review_vector = vectorizer.transform([new_review])
prediction = model.predict(new_review_vector)
if prediction == 1:
    print('Review is positive')
else:
    print('Review is negative')