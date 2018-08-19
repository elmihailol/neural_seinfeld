import pandas

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


data = pandas.read_csv("seinfeld_scripts/scripts.csv",
                       names=["id", "Character", "Dialogue", "EpisodeNo", "SEID", "Season"], keep_default_na=False)
print("Input character(GEORGE,ELAINE,KRAMER,JERRY): ")
ch = input()
data = data.query("Character == '"+ch+"'")
print((data['Character'].value_counts()))


Dialogue = data['Dialogue'].values.tolist()

model = load_model("conv1d.h5")
tokenizer = joblib.load("conv1d_tokenizer.sav")
le = joblib.load("conv1d_le.sav")

characters = le.classes_
print(characters)
pred_sum = numpy.zeros(len(characters))
for i in range(len(Dialogue)):
    speech = Dialogue[i]
    dataX = [tokenizer.texts_to_matrix(list(prepare_word(speech, n=33)))]
    dataX = numpy.array(dataX)
    pred = model.predict(dataX)
    pred_sum += pred[0]
    print("")
    for i in range(len(characters)):
        print(characters[i], "\t", pred[0][i], "\t", pred_sum[i])


