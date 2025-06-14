import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
english_stops = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in english_stops])
    return text

def load_tokenizer():
    with open('models/tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

def prepare_input(text, max_length=30):
    tokenizer = load_tokenizer()
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=max_length, padding='post')
