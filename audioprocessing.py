import pandas as panda;
import random;
from scipy.io import wavefile;
from pathlib import Path;


pathToSad = ("/data_folder/sad")
pathToHappy = ("/data_folder/happy")
pathToAngry = ("/data_folder/angry")
pathToFear = ("/data_folder/fear")

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
    if (x<=30 and sadNamesTest.__sizeof__<=30):
        sadNamesTest.append(sadAudio)
    else:
        sadNamesTrain.append(sadAudio)

for fearAudio in pathToFear.glob("*.wav"):
    x = random.randint(1,100)
    if (x<=30 and fearNamesTest.__sizeof__<=30):
        fearNamesTest.append(fearAudio)
    else:
        fearNamesTrain.append(fearAudio)

for happyAudio in pathToHappy.glob("*.wav"):
    x = random.randint(1,100)
    if (x<=30 and happyNamesTest.__sizeof__<=30):
        happyNamesTest.append(happyAudio)
    else:
        happyNamesTrain.append(happyAudio)

for angryAudio in pathToAngry.glob("*.wav"):
    x = random.randint(1,100)
    if (x<=30 and angryNamesTest.__sizeof__<=30):
        angryNamesTest.append(angryAudio)
    else:
        angryNamesTrain.append(angryAudio)
