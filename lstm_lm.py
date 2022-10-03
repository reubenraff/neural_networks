#!/usr/bin/env python
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Attention, Embedding, LSTM, Dense, Dropout, GRU, SimpleRNN, Bidirectional,Layer, Input
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np
from attention_layer import attention

#import attention layer

tokenizer = Tokenizer()


def dataset_preparation(data):
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1


    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        print(token_list)
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len,padding="pre"))


    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label,num_classes=total_words)

    return(predictors, label, max_sequence_len, total_words)

#def make_RNN(hidden_units, dense_units, input_shape, activation):
#def bidirectional_lstm(predictors,label,max_sequence_len, total_words):
def bidirectional_lstm(predictors,label,max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150,return_sequences=True)))
    model.add(attention(return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(total_words,activation="sigmoid"))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),optimizer="adam",metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="loss",min_delta=0,patience=5,verbose=0,mode="auto")
    model.fit(predictors,label,epochs=30,verbose=1,callbacks=[earlystop])
    print(model.summary())
    return(model)
#Is the man that is a man a man


def generate_text(seed_text,next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding="pre")
        predicted = model.predict_classes(token_list,verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return(seed_text)


with open("test_data.tok") as source:
        data = source.read()
        predictors, label, max_sequence_len, total_words = dataset_preparation(data)
        model = bidirectional_lstm(predictors,label,max_sequence_len, total_words)
        print(generate_text("Is the man",6,max_sequence_len))


#Is the man who is being kissed by his mother
#the grammatical RC is the predicted RC
