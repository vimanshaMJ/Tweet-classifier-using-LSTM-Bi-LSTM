import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
from utils import preprocess_text

# Load and preprocess data
train_df = pd.read_csv('data/phm_train.csv')
train_df['cleaned_text'] = train_df['tweet'].apply(preprocess_text)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['cleaned_text'])
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Padding
max_length = 30  # Set based on your data analysis
x_train = pad_sequences(
    tokenizer.texts_to_sequences(train_df['cleaned_text']),
    maxlen=max_length,
    padding='post'
)
y_train = train_df['label'].values

# Model building functions
def build_lstm(vocab_size=10000):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm(vocab_size=10000):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        Bidirectional(LSTM(64)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and save models
checkpoint = ModelCheckpoint('models/lstm_model.h5', save_best_only=True, monitor='val_accuracy')
lstm_model = build_lstm(len(tokenizer.word_index)+1)
lstm_model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[checkpoint])

checkpoint = ModelCheckpoint('models/bilstm_model.h5', save_best_only=True, monitor='val_accuracy')
bilstm_model = build_bilstm(len(tokenizer.word_index)+1)
bilstm_model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[checkpoint])
