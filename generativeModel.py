from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout,CuDNNLSTM
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import pandas as pd
import numpy as np
import string, os 
import matplotlib.pyplot as plt
from pickle import dump

data = pd.read_csv('sequences.csv')
corpus = data['Sequence'].astype(str).values.tolist()
#print(corpus[:10])

## tokenization
tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
#print(inp_sequences[:10])

def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, labels = input_sequences[:,:-1],input_sequences[:,-1]
    labels = ku.to_categorical(labels, num_classes=total_words)
    return predictors, labels, max_sequence_len

predictors, labels, max_sequence_len = generate_padded_sequences(inp_sequences)

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    #Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    # LSTM Layer
    model.add(CuDNNLSTM(100))
    model.add(Dropout(0.2))
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = create_model(max_sequence_len, total_words)
model.summary()

hist = model.fit(predictors, labels, epochs=20, validation_split=0.30,verbose=1)

hist_df = pd.DataFrame(hist.history)
plt.plot(hist_df['loss'])
plt.plot(hist_df['val_loss'])
#plt.title('model loss')

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()

# save the generative model to file, so can be used to generate sequences later on
model.save('generator.h5')

# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))

#Generating sequqnces
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

sequences=[]

tokens = pd.read_csv('tokens.csv')['TokenPair'].astype(str).values.tolist()
#print(tokens[:10])

#Generating synthetic sequences
for seedToken in tokens:
  generatedSeq = (generate_text(seedToken,1, model, max_sequence_len))
  if len(data[data['Sequence']==generatedSeq])==0:
    if generatedSeq not in sequences:
      sequences.append(generatedSeq)
    
#print(len(sequences))

# Saving File
with open('generated.txt', 'w') as f:
    for item in sequences:
        f.write("%s\n" % item)