import random

import numpy
import string
import pandas
import matplotlib.pyplot as plt
from keras import Input, Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from keras_preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

numpy.set_printoptions(threshold=numpy.nan)


def prepare_word(text, n=20):
    text = text.lower()
    text_len = len(text)
    while text_len < n:
        text += " "
        text_len = len(text)
    text = text[:n]
    return text


chars = []
chars.extend(list(string.ascii_lowercase))
# chars.extend(list("йцукенгшщззхъфывапролджэячсмитьбю1234567890+ "))
chars.extend(list(' '))
char_len = len(chars)
dataX = []
dataY = []
tokenizer_name = Tokenizer(filters='')
tokenizer_name.fit_on_texts(chars)

data = pandas.read_csv("seinfeld_scripts/scripts.csv",
                       names=["id", "Character", "Dialogue", "EpisodeNo", "SEID", "Season"], keep_default_na=False)
data = data.query("Character == 'JERRY' or Character == 'GEORGE' or Character == 'ELAINE' or Character == 'KRAMER'")
print((data['Character'].value_counts()))

character_dialog = {}

Character = data['Character'].values.tolist()
Dialogue = data['Dialogue'].values.tolist()

for i in range(len(Dialogue)):
    if Character[i] in character_dialog:
        character_dialog[Character[i]].append(Dialogue[i])
    else:
        character_dialog[Character[i]] = []
        character_dialog[Character[i]].append(Dialogue[i])

for key, value in character_dialog.items():
    random.shuffle(value, random.random)
    value = value[:6664]
    for i in range(len(value)):
        dataX.append(tokenizer_name.texts_to_matrix(list(prepare_word(value[i], n=100))))
        dataY.append(key)

c = list(zip(dataX, dataY))
random.shuffle(c)
dataX, dataY = zip(*c)

le = LabelBinarizer()
dataY = le.fit_transform(dataY)
dataX = numpy.array(dataX)
dataY = numpy.array(dataY)
model = Sequential()
model.add(Conv1D(128, 3, activation='relu', input_shape=(100, 27)))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(len(dataY[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(dataX, dataY,
                    batch_size=128,
                    epochs=25,
                    verbose=1, class_weight='auto', validation_split=0.8)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()