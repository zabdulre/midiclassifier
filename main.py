import argparse
import os
import music21
from sklearn import model_selection, naive_bayes, metrics
from collections import Counter
import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
from music21 import analysis
import torch
# read the files in to midi objects
# send the object into a processing function
# the processing function creates one vector per object
# get the vector from the processing function, and put the correct label
# pass it to sci kit learn

Labels = {"romantic": 1, "modern": 2, "classical": 3, "baroque": 4}
loadedData = []  # List of examples (vectors) for each era of music, each example is a dictionary {Label : feature vector}
loadedLabels = []
labelsToLoad = []
filesToLoad = []

#Number of valid files that had errors when analyzing/extracting features (note: does not account for files that were unable to be parsed in the first place)
badfilecount=0

#List of all key signatures
ksList = []
for a in ["c","d","e", "f", "g", "a", "b"]:
    for b in ["","-"]:
        for c in [" major", " minor"]:
            ksList.append(a+b+c)

#TODO
def getFeaturesFromMIDIObject(midiObject):
    badfileflag = 0
    features = []

    #Get Enumerated Key Signature
    #Can't seem to find "analyze" function
    #https://www.kaggle.com/code/wfaria/midi-music-data-extraction-using-music21
    try:
        ks = midiObject.analyze('key')
        ksIndex = ksList.index(str(ks).lower())
        print(ks, ksIndex)
        features.append(ksIndex)
    except Exception as e:
        print(e)
        badfileflag+=1
        features.append(-1)

    #Get min and max pitch (allegedly)
    #https://web.mit.edu/music21/doc/moduleReference/moduleAnalysisDiscrete.html
    try:
        p = analysis.discrete.Ambitus()
        pitchMin, pitchMax = p.getPitchSpan(midiObject)
        print(pitchMin.ps, pitchMax.ps)
        features.append(pitchMin.ps)
        features.append(pitchMax.ps)
    except Exception as e:
        print(e)
        badfileflag+=1
        features.append(-1)
        features.append(-1)

    if(badfileflag!=0):
        global badfilecount
        badfilecount +=1
        print(badfileflag, "errors when parsing features") 
    
    print(features)
    return features

def updateLoadedData(pos, length, out, result):
    if out is not None:
        #if len(labelsToLoad) > pos: #only add to the back of the list, prevents duplicates
        if pos >= len(loadedData):
            print("Parsed file number ", pos + 1, " ", result)
            loadedData.append(getFeaturesFromMIDIObject(out))
            loadedLabels.append(labelsToLoad[pos])
    return

def tryToParse(filePath):
    try:
        return music21.converter.parse(filePath)
    except:
        print(filePath, "could not be parsed, skipping...")
        return None

def getFeatures():
    # use music21 to open the file
    music21.common.runParallel(filesToLoad, tryToParse, updateMultiply=1, updateFunction=updateLoadedData, updateSendsIterable=True)
    #file = music21.converter.parse(filePath)
    #print(file.flat.keySignature.asKey())
    # get a vector for each feature we want to add
    # append those vectors together into one giant vector
    global badfilecount
    print("total bad files: ", badfilecount)
    print("Done!")
    return loadedData.copy(), loadedLabels.copy()


def getClassDirectories(rootdir, argv):
    """
    @param rootdir: which directory to look for the era folders in
    @param argv: Contains the datadir and the name of each folder in the datadir. Each folder corresponds to an era of music.
    @return: A dictionary of directories. Each directory corresponds to a genre of music. Inside the directory should be MIDI files.
    """
    return {"romantic": rootdir + "/" + argv.romantic, "baroque": rootdir + "/" + argv.baroque,
            "classical": rootdir + "/" + argv.classical, "modern": rootdir + "/" + argv.modern}


def loadMIDIs(directories):
    """
    @param directories: a list of all directories to extract midis from (subfolders must be a valid era name)
    @return X, Y: features and labels respectively for the MIDI files
    """
    labelsToLoad.clear()
    loadedData.clear()
    loadedLabels.clear()
    filesToLoad.clear()

    for folder in directories.items():

        #Debug: only run on one folder
        if(folder[0]!="modern"):
            continue

        currentLabel = Labels[folder[0]]
        for fileName in os.listdir(folder[1]):
            if fileName[0] == '.':  # skip any hidden files
                continue
            currentFileDirectory = folder[1] + "/" + fileName
            filesToLoad.append(currentFileDirectory)
            labelsToLoad.append(currentLabel)

    X, Y = getFeatures()
    return X, Y

def printDatasetMetrics(X, Y, title="Training Dataset"):
    """
    @param X: list of feature vectors
    @param Y: list of corresponding label vectors
    """
    if X is not None:
        pass #print feature related metrics here
    if Y is not None and len(Y) != 0:
        f, aa = plt.subplots()
        plt.title("Label Distribution for " + title)
        aa.pie(Counter(Y).values(), labels=list(Labels.keys())[:1], autopct='%1.0f%%') #change the labels according to if the dataset has all eras or not TODO
        plt.show()
        pass #print label related metrics here
    if X is not None and Y is not None:
        pass #print metrics related to both here


def doNaiveBayes(X_train, X_test, Y_train, Y_test):
    nb = naive_bayes.GaussianNB()
    nb.fit(X_train, Y_train)
    metrics(nb, X_test, Y_test)
    print(metrics.classification_report(Y_test, nb.predict(X_test)))

def doMLP(X_train, X_test, Y_train, Y_test):


def main(argv):
    trainDirectories = getClassDirectories(argv.datadir, argv)
    X, Y = loadMIDIs(trainDirectories)
    printDatasetMetrics(X, Y)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25) #can replace this by loading the test set instead
    doNaiveBayes(X_train, X_test, Y_train, Y_test)
    doMLP(X_train, X_test, Y_train, Y_test)
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=str, help="directory to folders of midi files", required=True)
    p.add_argument("--romantic", default="romantic", type=str, help="name of folder of romantic era midi files")
    p.add_argument("--baroque", default="baroque", type=str, help="name of folder of baroque era midi files")
    p.add_argument("--classical", default="classical", type=str, help="name of folder of classical era midi files")
    p.add_argument("--modern", default="modern", type=str, help="name of folder of modern/contemporary era midi files")
    arg = p.parse_args()
    main(arg)
