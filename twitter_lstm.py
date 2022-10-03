import keras
import numpy
import pandas as pd
import keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.layers import LSTM, GlobalMaxPool1D, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer


max_length = 128
vocab_size = 25000
emb_dim = 50
optimizer = "adam"
epochs = 10


# multilabel classification, each label is a 3dim vector
# 1 corresponds to sentiment present, 0 for sentiment absent

label_map = {
    "neutral": np.asarray([1,0,0]),
    "positive": np.asarray([0,1,0]),
    "negative": np.asarray([0,0,1])
}

tweets = pd.read_csv('twitter_data.csv').sample(frac=1)


def tweet_lstm():

    num_train = round(len(tweets) * 0.85) #this is val data
    num_test = len(tweets) - num_train
    text = tweets["test"].tolist()
    labels = tweets["airline_sentiment"].tolist()
    labels = list(map(lambda label: label_map[label], labels))
    text_train = text[0:num_train]
    text_test = text[num_train:]
    labels_train = np.asarray(labels[0:num_train])
    labels_test = labels[num_train:]

    tokenizer = Tokenizer(num_words=vocab_size,oov_token="[UNK]")
    tokenizer.fit_on_texts(text_train)

    sequences_train = tokenizer.texts_to_sequences(text_train)
    sequences_test = tokenizer.texts_to_sequences(text_test)

    #fit text to lists  of ints then pad them so they are all the
    #same length

    padded_train = pad_sequences(sequences_train, maxlen=max_length,padding="post")

    padded_test = pad_sequences(sequences_test, maxlen=max_length, padding="post")

    model = Sequential()
    model.Embedding(vocab_size, emb_dim, input_length=max_length)
    model.add(Bidirectional(LSTM(20,return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.compile(optimizer=optimizer,loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])

    model.fit(padded_train, labels_train, validation_data=(padded_test, sequences_test),epochs=epochs,validation=0.1)

tweet_lstm()
