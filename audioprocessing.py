import pandas as panda;
import random;
import librosa;
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

#Exploratory Data Analysis

#Using librosa
sad, sad2 = librosa.load(sadNamesTest[0])
print(type(sad))
print(type(sad2))