import numpy as np
import pandas as pd
import os

from preprocessing import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle

def load_data():
        filename = input("Enter exact file name: ")
        name, extension = filename.split(".csv")

        if os.path.exists(name + 'preprocessed.csv'):
            print("Found the preprocessed data.", len(name+'preprocessed.csv'))
            bool = input("Do you want a repeated preprocess?\n")
            if bool.lower() == 'y' or bool.lower() == 'yes': 
                df['preprocessed_text'] = df['clean_text'].apply(preprocess_text)
                df.to_csv(name + 'preprocessed.csv', index=False)
            else: 
                df = pd.read_csv(name + 'preprocessed.csv')
        
        else:
            df = pd.read_csv(name + '.csv')
            df['preprocessed_text'] = df['clean_text'].apply(preprocess_text)
            df.to_csv(name + 'preprocessed.csv', index=False)
            print("Complete the preprocessing and made the csv", len(name+'preprocessed.csv'))

        return df
    
def split_data(df):
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=None)

    # Training data
    X_train = train_df['preprocessed_text']
    y_train = train_df['category']

    # Testing data
    X_test = test_df['preprocessed_text']
    y_test = test_df['category']

    return X_train, X_test, y_train, y_test

def main():
    df = load_data()
    # Model training
    # Call the functions or classes to train the sentiment analysis model
    
    df.isnull().sum()
    df = df.dropna()
    df = df.reset_index(drop=True)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Performing feature extraction using CountVectorizer...") 
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    classifier = LogisticRegression(C=1.0, penalty='l1', max_iter=10000, solver='liblinear')

    X_train_bow = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_bow, y_train)

    # Evaluate the model on the test set
    print("Evaluating the model with test data...")
    X_test_bow = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_bow)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy: ", accuracy)

    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 score: ", f1)

    with open('model.pkl', 'wb') as file: 
        pickle.dump(classifier, file)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == '__main__':
    main()
