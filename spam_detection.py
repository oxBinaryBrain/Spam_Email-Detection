
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load data from CSV file
data = pd.read_csv('spam_dataset.csv')

# Assuming your CSV has two columns: 'text' for email content and 'label' for spam or not spam
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the emails using Bag-of-Words representation
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Predictions
predictions = clf.predict(X_test_counts)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Test with new emails
new_emails = [
    "Free offer! Click now to win a holiday.",
    "Meeting rescheduled to tomorrow.",
]
new_emails_counts = vectorizer.transform(new_emails)
new_predictions = clf.predict(new_emails_counts)

for email, prediction in zip(new_emails, new_predictions):
    if prediction == 1:
        print(f"'{email}' is classified as SPAM.")
    else:
        print(f"'{email}' is classified as NOT SPAM.")
