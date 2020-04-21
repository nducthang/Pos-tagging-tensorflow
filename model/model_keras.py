import tensorflow as tf
import process_data
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, TimeDistributed
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    path = '../data/train.txt'
    MAX_LENGTH = 120
    sentences = process_data.load_data(path)
    word2idx, idx2word, tag2idx, idx2tag = process_data.parse(sentences)
    X_train, y_train = process_data.sentence_to_number(sentences, MAX_LENGTH, word2idx, tag2idx)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    model = Sequential()
    model.add(Embedding(input_dim=len(word2idx.items()), output_dim=100, input_length=MAX_LENGTH))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1)))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.1)))
    model.add(Dense(len(tag2idx.items()), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    BATCH_SIZE = 128
    EPOCHS = 15

    checkpoint = ModelCheckpoint(
        'model_keras.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True)

    history = model.fit(
        X_train,
        np.array(y_train),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=[X_test, np.array(y_test)],
        verbose=1,
        callbacks=[checkpoint]
    )