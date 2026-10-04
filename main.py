import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense,Dropout
from keras.models import Sequential

fakeCSV = pd.read_csv('Fake.csv')
trueCSV = pd.read_csv('True.csv')

fakeCSV['label'] = 0
trueCSV['label'] = 1

trueCSVFixedText = trueCSV['text'].str.split(' - ', n=1).str[-1]

fakeCSV['total_text'] = fakeCSV['title'].fillna('') + ' ' + fakeCSV['text'].fillna('')
fakeCSV['total_text']=fakeCSV['total_text'].str.lower().str.replace(r'[^a-z\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
trueCSV['total_text'] = trueCSV['title'].fillna('') + ' ' + trueCSVFixedText.fillna('')
trueCSV['total_text']=trueCSV['total_text'].str.lower().str.replace(r'[^a-z\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

dataset = pd.concat([fakeCSV[['total_text', 'label']], trueCSV[['total_text', 'label']]], axis=0)

dataset = dataset.sample(frac=1, random_state=5, ignore_index=True)

trainSize=int(len(dataset)*0.8)
train=dataset.iloc[:trainSize]
test=dataset.iloc[trainSize:]
xTrain,yTrain=train['total_text'],train['label']
xTest,yTest=test['total_text'],test['label']

#-------------------------------------------------------------

textVector=TextVectorization(max_tokens=10000,output_mode='int',output_sequence_length=400)
textVector.adapt(xTrain.to_numpy())

#-------------------------------------------------------------

model = Sequential()
model.add(textVector)
model.add(Embedding(input_dim=10000, output_dim=64))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(xTrain.to_numpy(),yTrain.to_numpy(),epochs=2,batch_size=32,validation_data=(xTest.to_numpy(), yTest.to_numpy()))
model.save("fake_news_detector.keras")
print("\nModel is saved!")
