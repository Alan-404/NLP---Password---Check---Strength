#%%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv('../datasets/data.csv')
# %%
df.describe()
# %%
df.info()
# %%
df.head(10)
# %%
df['strength'].value_counts()
# %%
df['password'][df['password'] == "selim"]
# %%
df[df['1'].isna() == False]
# %%
def check_strength(item):
    if item == '0' or item == '1' or item == '2':
        return True
    return False
# %%
df['check'] = df['strength'].apply(lambda item: check_strength(item))
# %%
df['check'].value_counts()
# %%
df['password'] = df['password'].fillna("")

# %%

# %%

#%%
df = df[['password', 'strength']][df['check'] == True]
# %%
df['strength'].value_counts()
# %%
def process_password(password: str):
    lst = []
    for item in password:
        
        lst.append(str(item))
    return " ".join(lst)
# %%
df['password_processed'] = df['password'].apply(lambda item:process_password(item))
# %%
df['password_processed']
# %%
df['strength'] = df['strength'].apply(lambda item:int(item))
# %%
from keras.preprocessing.text import Tokenizer
# %%
X = np.array(df['password_processed'])
y = np.array(df['strength'])
# %%
X
# %%
tokenizer = Tokenizer()
# %%
tokenizer.fit_on_texts(X)
#%%
import pickle
import io
# %%
with io.open('../tokenizer/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%

# %%

# %%
tokenizer.word_index
# %%
sequences = tokenizer.texts_to_sequences(X)
# %%
from keras_preprocessing.sequence import pad_sequences
# %%
max_length = 50
X_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
# %%
X_sequences.shape
# %%
y.shape
# %%
X_sequences[0]
#%%
len(tokenizer.word_counts)
# %%
len(tokenizer.word_index)
# %%

# %%

# %%
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, LayerNormalization
# %%
model = Sequential()
# %%
embedding_dim = 64
model.add(Embedding(input_dim=len(tokenizer.word_counts)+1, output_dim=embedding_dim,input_length=max_length))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# %%
model.fit(X_sequences, y, epochs=5)
# %%
model.save('../saved_models/first.h5')
# %%

# %%

# %%

# %%
df