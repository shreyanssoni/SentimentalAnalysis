import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess_text
import pandas as pd

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict_sentiment(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    text_vector = vectorizer.transform([preprocessed_text])

    # Make the prediction
    prediction = model.predict(text_vector)[0]

    return prediction

# Function to collect user data and labels
def collect_user_data():
    user_data = []
    user_labels = []
    while True:
        data = input("Enter the text data (or 'q' to quit): ")
        if data == 'q':
            break
        preprocess_text(data)
        print(predict_sentiment(data))
        label = input("Enter the corresponding label: ")
        user_data.append(data)
        user_labels.append(label)
    return user_data, user_labels

# Load the existing dataset
existing_data = pd.read_csv('shuffled_datapreprocessed.csv')
print(len(existing_data))

# Collect user data and labels
user_data, user_labels = collect_user_data()

# Create a DataFrame for user data
user_df = pd.DataFrame({'clean_text': user_data, 'preprocessed_text': user_data, 'category': user_labels})

# Concatenate user data with existing data
combined_data = pd.concat([existing_data, user_df], ignore_index=True)

# Save the combined data to a CSV file
combined_data.to_csv('combined_data.csv', index=False)
