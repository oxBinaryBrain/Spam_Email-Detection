# Spam Email Detection using scikit-learn

This project demonstrates how to build a simple spam email detection system using scikit-learn, a popular machine learning library in Python.

## Overview

The project uses a Bag-of-Words model and the Naive Bayes classifier to classify emails as spam or not spam. It includes the following components:

1. Loading the dataset from a CSV file.
2. Preprocessing the data and splitting it into training and testing sets.
3. Vectorizing the emails using the Bag-of-Words representation.
4. Training a Naive Bayes classifier on the training data.
5. Evaluating the model's accuracy on the testing data.
6. Making predictions on new emails.

## Requirements

- Python 3.x
- scikit-learn
- pandas

## Usage

1. Ensure you have Python installed on your system.
2. Install the required libraries using pip:
3. Download the `spam_dataset.csv` file or prepare your own dataset in a similar format.
4. Run the provided Python script `spam_detection.py`.
5. The script will train the model, evaluate its accuracy, and make predictions on new emails.

## Dataset

The dataset used in this project is stored in a CSV file (`spam_dataset.csv`). It contains two columns: 'text' for the email content and 'label' indicating whether each email is spam (1) or not spam (0). You can replace this dataset with your own CSV file following the same format.

## Additional Notes

- The code provided here is a basic example. For better accuracy, you may consider using more advanced techniques, such as feature engineering, hyperparameter tuning, or using more sophisticated classifiers.
- Ensure that your dataset is well-balanced and representative to build a robust spam detection model.
- Experiment with different vectorization techniques and classifiers to find the best combination for your specific use case.

