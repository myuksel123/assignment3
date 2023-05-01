import pandas as panda;
import random;
import librosa;
import numpy as np
import matplotlib.pyplot as plot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
#from scipy.io import wavefile;
from pathlib import Path;

#sorting them into train and test datasets
#randomly put into each

pathToSad = Path("./data_folder/sad")
pathToHappy = Path("./data_folder/happy")
pathToAngry = Path("./data_folder/angry")
pathToFear = Path("./data_folder/fear")

sadNamesTest = []
sadNamesTrain = []

happyNamesTest =[]
happyNamesTrain=[]

angryNamesTest = []
angryNamesTrain=[]

fearNamesTest=[]
fearNamesTrain=[]

for sadAudio in pathToSad.glob("*.wav"):
    x = random.randint(1,100)
    if (x<=30 and len(sadNamesTest)<=30):
        sadNamesTest.append(str(sadAudio))
    else:
        sadNamesTrain.append(str(sadAudio))

for fearAudio in pathToFear.glob("*.wav"):
    x = random.randint(1,100)
    if (x<=30 and len(fearNamesTest)<=30):
        fearNamesTest.append(str(fearAudio))
    else:
        fearNamesTrain.append(str(fearAudio))

for happyAudio in pathToHappy.glob("*.wav"):
    x = random.randint(1,100)
    if (x<=30 and len(happyNamesTest)<=30):
        happyNamesTest.append(str(happyAudio))
    else:
        happyNamesTrain.append(str(happyAudio))

for angryAudio in pathToAngry.glob("*.wav"):
    x = random.randint(1,100)
    if (x<=30 and len(angryNamesTest)<=30):
        angryNamesTest.append(str(angryAudio))
    else:
        angryNamesTrain.append(str(angryAudio))


#put features and their respective moods in a dataframe

extracted_Train =[]

#putting sad files
for sadA in sadNamesTrain:
    sad, sadsr = librosa.load(sadA)
    mfccs = librosa.feature.mfcc(y=sad, sr=sadsr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    extracted_Train.append([mfccs_scaled, "sad"])

#putting happy files into train 
for audio in happyNamesTrain:
    featuresA, srA = librosa.load(audio)
    mfccs = librosa.feature.mfcc(y=featuresA, sr=srA, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    extracted_Train.append([mfccs_scaled, "happy"])

#putting fear audio files
for audio in fearNamesTrain:
    featuresA, srA = librosa.load(audio)
    mfccs = librosa.feature.mfcc(y=featuresA, sr=srA, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    extracted_Train.append([mfccs_scaled, "fear"])

#putting angry audio files
for audio in fearNamesTrain:
    featuresA, srA = librosa.load(audio)
    mfccs = librosa.feature.mfcc(y=featuresA, sr=srA, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    extracted_Train.append([mfccs_scaled, "angry"])


trainingDF = panda.DataFrame(extracted_Train, columns = ["features", "mood"])
print(trainingDF.head())




#lets also prepare our test data

extracted_Test =[]

#putting sad files
for sadA in sadNamesTest:
    sad, sadsr = librosa.load(sadA)
    mfccs = librosa.feature.mfcc(y=sad, sr=sadsr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    extracted_Test.append([mfccs_scaled, "sad"])

#putting happy files into train 
for audio in happyNamesTest:
    featuresA, srA = librosa.load(audio)
    mfccs = librosa.feature.mfcc(y=featuresA, sr=srA, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    extracted_Test.append([mfccs_scaled, "happy"])

#putting fear audio files
for audio in fearNamesTest:
    featuresA, srA = librosa.load(audio)
    mfccs = librosa.feature.mfcc(y=featuresA, sr=srA, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    extracted_Test.append([mfccs_scaled, "fear"])

#putting angry audio files
for audio in fearNamesTest:
    featuresA, srA = librosa.load(audio)
    mfccs = librosa.feature.mfcc(y=featuresA, sr=srA, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    extracted_Test.append([mfccs_scaled, "angry"])


testingDF = panda.DataFrame(extracted_Test, columns = ["features", "mood"])
print(testingDF.head())

#X will be the features, y will be their categorical mood
trainX = np.array(trainingDF['features'].tolist())
trainY = np.array(trainingDF['mood'].tolist())
trainY = to_categorical(LabelEncoder().fit_transform(trainY))

testX = np.array(testingDF['features'].tolist())
testY = np.array(testingDF['mood'].tolist())
testY = to_categorical(LabelEncoder().fit_transform(testY))


#Creating the model, based on a tutorial I found on creating models with keras models
audioModel = Sequential()
audioModel.add(Dense(100,input_shape=(40,)))
audioModel.add(Activation('relu'))
audioModel.add(Dropout(0.5))

audioModel.add(Dense(200))
audioModel.add(Activation('relu'))
audioModel.add(Dropout(0.5))

audioModel.add(Dense(100))
audioModel.add(Activation('relu'))
audioModel.add(Dropout(0.5))


audioModel.add(Dense(4))
audioModel.add(Activation('softmax'))

#compiling the model, based on a tutorial
audioModel.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

audioModel.fit(trainX, trainY, batch_size=32, epochs=10, validation_data=(testX, testY), verbose=1)

#testing the model
accuracy = audioModel.evaluate(testX, testY, verbose = 0)

print("The accuracy of the model is "+ str(accuracy[1]*100) + "%")