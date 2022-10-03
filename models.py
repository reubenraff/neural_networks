
def base_model():
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(LSTM(150,return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(total_words,activation="sigmoid"))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),optimizer="adam",metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="loss",min_delta=0,patience=5,verbose=0,mode="auto")
    model.fit(predictors,label,epochs=30,verbose=1,callbacks=[earlystop])
    print(model.summary())
    return(model)
base_model()

"""prediction 6 tokens:  Is the man who is being kissed by his mother
   prediction 8 tokens: Is the man who is on the chair is happy happy happy
 """



    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(SimpleRNN(150,return_sequences=True))
    model.add(SimpleRNN(100))
    model.add(Dense(total_words,activation="sigmoid"))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),optimizer="adam",metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="loss",min_delta=0,patience=5,verbose=0,mode="auto")
    model.fit(predictors,label,epochs=30,verbose=1,callbacks=[earlystop])
    print(model.summary())
    return(model)

"""prediction: Is the man who is stealing a drug dealer target""""





def gated():
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(GRU(150,return_sequences=True))
    model.add(GRU(100))
    model.add(Dense(total_words,activation="sigmoid"))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),optimizer="adam",metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="loss",min_delta=0,patience=5,verbose=0,mode="auto")
    model.fit(predictors,label,epochs=30,verbose=1,callbacks=[earlystop])
    print(model.summary())
    return(model)



def bidirectional_lstm():
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150,return_sequences=True)))
    model.add(LSTM(100))
    model.add(Attention(use_scale=False))
    model.add(Dense(total_words,activation="sigmoid"))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),optimizer="adam",metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="loss",min_delta=0,patience=5,verbose=0,mode="auto")
    model.fit(predictors,label,epochs=30,verbose=1,callbacks=[earlystop])
    print(model.summary())
    return(model)

"""
8 token prediction: Is the man who is on the chair his mother mother happy
7 token prediction: Is the man who is being kissed by his mother
6 token prediction: Is the man who is on the chair black mother
"""

def bidirectional_lstm():
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150,return_sequences=True)))
    model.add(LSTM(100))
    model.add(Dense(total_words,activation="sigmoid"))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),optimizer="adam",metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="loss",min_delta=0,patience=5,verbose=0,mode="auto")
    model.fit(predictors,label,epochs=30,verbose=1,callbacks=[earlystop])
    print(model.summary())
    return(model)



"""
prediction with attention: Is the man that is is mouse is is on
"""


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



"""
disambiguation

SRN / transformer
"""
