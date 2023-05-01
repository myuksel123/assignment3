import pandas as panda;
import random;
import librosa;
import numpy as np
import matplotlib.pyplot as plot
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

#X will be the features, y will be their categorical mood
trainX = np.array(trainingDF['features'].tolist())
trainY = np.array(trainingDF['mood'].tolist())