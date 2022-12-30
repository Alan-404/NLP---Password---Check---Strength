#%%
from keras.models import load_model
import pickle
from keras_preprocessing.sequence import pad_sequences
# %%
with open('../tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# %%
model = load_model('../saved_models/first.h5')
# %%
max_length = 50
# %%
def preprocessing(sequence: str):
    lst = []
    for item in sequence:
        lst.append(item)
    sentence = " ".join(lst)
    input_pred = tokenizer.texts_to_sequences([sentence])
    return pad_sequences(input_pred, maxlen=max_length, padding='post', truncating='post')
# %%
a = preprocessing("123")
# %%
result = model.predict(a)
# %%
result.argmax()
# %%
