import numpy

import joblib
from keras.engine.saving import load_model


def prepare_word(text, n=20):
    text = text.lower()
    text_len = len(text)
    while text_len < n:
        text += " "
        text_len = len(text)
    text = text[:n]
    return text


model = load_model("conv1d.h5")
tokenizer = joblib.load("conv1d_tokenizer.sav")
le = joblib.load("conv1d_le.sav")

characters = le.classes_
print(characters)
pred_sum = numpy.zeros(len(characters))
while 1:
    print("Your speech: ")
    speech = input()
    dataX = [tokenizer.texts_to_matrix(list(prepare_word(speech, n=33)))]
    dataX = numpy.array(dataX)
    pred = model.predict(dataX)
    pred_sum += pred[0]
    for i in range(len(characters)):
        print(characters[i], "\t", pred[0][i], "\t", pred_sum[i])


