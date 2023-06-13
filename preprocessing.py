import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    try:
        if pd.isnull(text):
            return ''

        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        tokens = word_tokenize(text)

        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]

        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        preprocessed_text = ' '.join(lemmatized_tokens)

        return preprocessed_text
    
    except Exception as e:
        print("Error occurred: ", str(e))
        return ''
