# Email Urgency Classifier

A machine learning project that classifies emails into priority levels and assigns a continuous urgency score using natural language processing techniques.

## Overview

This project uses a TF-IDF vectorization pipeline combined with a multinomial Logistic Regression model to analyze email text and predict its urgency. The model outputs both a discrete priority label (Low, Medium, High) and a continuous urgency score from 0–100 derived from class probabilities.

The goal of this project is to demonstrate applied NLP, supervised learning, and clean machine learning project structure.

## Features

- Text preprocessing with TF-IDF (unigrams and bigrams)
- Multiclass classification using Logistic Regression
- Probability-based urgency scoring
- Train/test evaluation with precision, recall, and F1 metrics
- Modular, production-style Python code

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TF-IDF (Natural Language Processing)

## Dataset

The model is trained on a CSV dataset containing labeled email messages.

Expected columns:
- `text`: email content
- `label`: priority level  
  - 0 = Low  
  - 1 = Medium  
  - 2 = High  

Dataset file:
`emails_priority.csv`

Install dependencies:
pip install -r requirements.txt

Usage
Run the training and evaluation script:
python src/email_urgency.py

The script will:
- Train the model
- Print a classification report
- Generate sample urgency predictions

Sample Output
Email: Your tuition payment is overdue
Predicted Priority: High (2)
Urgency Score: 92.5/100
##
