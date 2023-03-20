import argparse
import os
import music21
import torch
import pandas as pd
from sklearn import model_selection, naive_bayes, metrics
from collections import Counter
import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
from music21 import analysis
from torch import tensor, nn, optim, no_grad, argmax, sum
from torch.utils.data import DataLoader, dataset
from music21 import pitch
from Model import MLPmodel
from tqdm import tqdm

# read the files in to midi objects
# send the object into a processing function
# the processing function creates one vector per object
# get the vector from the processing function, and put the correct label
# pass it to sci kit learn

Labels = {"romantic": 0, "modern": 1, "classical": 2, "baroque": 3}
loadedData = []  # List of examples (vectors) for each era of music, each example is a dictionary {Label : feature vector}
loadedLabels = []
labelsToLoad = []
filesToLoad = []

# Number of valid files that had errors when analyzing/extracting features (note: does not account for files that were unable to be parsed in the first place)
badfilecount = 0
numF = 0
# List of all key signatures
# ksList = []
# for a in ["c", "d", "e", "f", "g", "a", "b"]:
#     for b in ["", "#", "-"]:
#         for c in [" major", " minor"]:
#             ksList.append(a + b + c)


# TODO
def getFeaturesFromMIDIObject(midiObject):
    badfileflag = 0
    features = []

    # Get Enumerated Key Signature: (NOTE: CHANGED to major/minor, sharpflat/not)
    # https://www.kaggle.com/code/wfaria/midi-music-data-extraction-using-music21
    # https://web.mit.edu/music21/doc/moduleReference/moduleKey.html
    # Analyze function documentation:
    # https://web.mit.edu/music21/doc/moduleReference/moduleStreamBase.html
    try:
        ks = midiObject.analyze('key')
        # Either use enumerated for every single key signature
        # ksIndex = ksList.index(str(ks).lower())
        # Or use number of sharps/flats (however, not too much relation between similar valued ks, and no way to do exception value?
        # ksIndex = ks.sharps
        # print(ks, '(',ksIndex,')')
        # features.append(ksIndex)

        # Change: now just have 2 features: boolean for major (1) or minor (0), and boolean for sharp/flat key sig (1) or not (0).
        ksString = str(ks).lower()
        if "major" in ksString:
            features.append(1)
        else:
            features.append(0)
        
        if ("#" in ksString) or ("-" in ksString):
            features.append(1)
        else:
            features.append(0)
    except Exception as e:
        print(e)
        badfileflag += 1
        # features.append(-1)
        features.append(-1)
        features.append(-1)

    # Get min and max pitch (allegedly)
    # https://web.mit.edu/music21/doc/moduleReference/moduleAnalysisDiscrete.html
    try:
        p = analysis.discrete.Ambitus()
        pitchMin, pitchMax = p.getPitchSpan(midiObject)
        # print(pitchMin.ps, pitchMax.ps)
        features.append(pitchMin.ps)
        features.append(pitchMax.ps)
    except Exception as e:
        print(e)
        badfileflag += 1
        features.append(-1)
        features.append(-1)

    # Get most common pitch (not considering octaves?, so pitchClass)
    # https://web.mit.edu/music21/doc/moduleReference/moduleAnalysisPitchAnalysis.html
    # https://web.mit.edu/music21/doc/moduleReference/modulePitch.html
    try:
        pitchClassCount = analysis.pitchAnalysis.pitchAttributeCount(midiObject, 'pitchClass')
        for n, count in pitchClassCount.most_common(3):
            # print("%2s: %d" % (pitch.Pitch(n), pitchClassCount[n]))
            features.append(n)
    except Exception as e:
        print(e)
        badfileflag += 1
        for n in range(3):
            features.append(-1)

    # For pitch, split into lower half and upper half, to simulate Left Hand/Right Hand
    try:
        psCount = analysis.pitchAnalysis.pitchAttributeCount(midiObject, 'ps')
        psList = sorted(psCount.elements())
        psLeft = psList[:len(psList) // 2]
        psRight = psList[len(psList) // 2:]

        # Get average pitch for each half
        apLeft = np.mean(psLeft)
        apRight = np.mean(psRight)
        # print(apLeft, apRight)
        features.append(apLeft)
        features.append(apRight)
        # Get standard deviation for each half
        sdpLeft = np.std(psLeft)
        sdpRight = np.std(psRight)
        # print(sdpLeft, sdpRight)
        features.append(sdpLeft)
        features.append(sdpRight)
    except Exception as e:
        print(e)
        badfileflag += 1
        for n in range(4):
            features.append(-1)

    # Other features to add: chords, intervals, tempo of song

    #Next feature: get rhythm features (average note length, note density, frequency of quarter, eighth, 16th notes)
    #Adapted from: https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1143&context=scschcomdis (on how to get all note durations)
    notesDurList = []
    try:
        for n in midiObject.flat.notes:
            if n.duration.isGrace is False and n.quarterLength != 0:
                notesDurList.append(float(n.quarterLength))
        #mean note duration
        durmean = np.mean(notesDurList)
        noteDens =  len(notesDurList) / float(midiObject.duration.quarterLength)

        numQuart = 0
        numEighth = 0
        numSixteenth = 0
        numTriplet = 0
        for n in notesDurList:
            if n == 1.0:
                numQuart+=1
            if n == 0.5:
                numEighth+=1
            if n == 0.25:
                numSixteenth+=1
            if n > 0.3 and n < 0.35:
                numTriplet+=1

        features.append(durmean)
        features.append(noteDens)

        features.append(numQuart/len(notesDurList))
        features.append(numEighth/len(notesDurList))
        features.append(numSixteenth/len(notesDurList))
        features.append(numTriplet/len(notesDurList))

    except Exception as e:
        print(e)
        badfileflag += 1
        for i in range(6):
            features.append(-1)


    # When moving to from baseline to LSTM/Transformer: do this for every couple measures?
    # How do we try analyzing sequence of smaller sections instead of just collecting overall statistics?
    # Maybe try to translate/convert/transform each 2-3 measures into a d-dimensional "word"?

    # If there were any errors, make note of parsing problem
    if (badfileflag != 0):
        global badfilecount
        badfilecount += 1
        print(badfileflag, "errors when parsing features")

        # Print and return features
    # Feature 1,2: major/major, keysig is # or - (or not) (boolean)
    # Features 3,4: max and min pitch. (numerical, discrete)
    # Features 5-7: "enumerated" 3 most common notes (not considering octave, so just note name) (categorical? discrete)
    # Features 8-9: Avg mean of upper half and lower half of notes (to simulate LH RH) (continuous)
    # Features 10-11: Std dev of upper half and lower half of notes (continuous)
    # Features 12,13: average duration of a note, average number of notes played per quarter note
    # Features 14-17: relative frequency of quarter, eighth, sixteenth, and triplet notes.
    print(features)
    return features


def updateLoadedData(pos, length, out, result):
    if out is not None:
        # if len(labelsToLoad) > pos: #only add to the back of the list, prevents duplicates
        if pos >= len(loadedData):
            print("Parsed file number ", pos + 1, " ", result)
            features = getFeaturesFromMIDIObject(out)
            global numF
            numF = len(features)
            loadedData.append(features)
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
    music21.common.runParallel(filesToLoad, tryToParse, updateMultiply=1, updateFunction=updateLoadedData,
                               updateSendsIterable=True)
    # file = music21.converter.parse(filePath)
    # print(file.flat.keySignature.asKey())
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

        # Debug: only run on one folder
        #if (folder[0] != "modern"):
        #    continue

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
        pass  # print feature related metrics here
    if Y is not None and len(Y) != 0:
        f, aa = plt.subplots()
        plt.title("Label Distribution for " + title)
        aa.pie(Counter(Y).values(), labels=list(Counter(Y).keys()),
               autopct='%1.0f%%')  # change the labels according to if the dataset has all eras or not TODO
        plt.show()
        pass  # print label related metrics here
    if X is not None and Y is not None:
        pass  # print metrics related to both here


def doNaiveBayes(X_train, X_test, Y_train, Y_test):
    nb = naive_bayes.GaussianNB()
    nb.fit(X_train, Y_train)
    print(metrics.classification_report(Y_test, nb.predict(X_test)))


def mlp_loaders(X_train, X_test, Y_train, Y_test, argv):
    trainDataset = dataset.TensorDataset(tensor(X_train,dtype=torch.float32), tensor(Y_train,dtype=torch.float32))
    testDataset = dataset.TensorDataset(tensor(X_test,dtype=torch.float32), tensor(Y_test,dtype=torch.float32))
    return DataLoader(trainDataset, argv.batchsize, shuffle=True), DataLoader(testDataset, argv.batchsize,
                                                                               shuffle=True)


def mlp_epoch(model, lossFunction, opt, loader, argv, train=True):
    numCorrect = 0
    numExamples = 0
    totalLoss = 0
    for (x, y) in loader:
        opt.zero_grad()
        modelOutput = model(x)
        loss = lossFunction(modelOutput, y.to(torch.long))
        if train:
            loss.backward()
            opt.step()

        predictions = argmax(modelOutput, 1)
        numCorrect += torch.sum(predictions == y).item()
        numExamples += len(y)
        totalLoss += loss.item()

    return numCorrect / numExamples, totalLoss


# This function is inspired by code from homework 2 of the csci467 course at USC, by Robin Jia
# No code is copied directly, but the same technique is used to run the model
def doMLP(X_train, X_test, Y_train, Y_test, argv):
    trainLoader, devLoader = mlp_loaders(X_train, X_test, Y_train, Y_test, argv)
    model = MLPmodel(len(Labels.keys()), numF, argv)
    lossFunction = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), argv.learning_rate)

    for i in tqdm(range(argv.epochs)):
        model.train()
        trainAcc, trainLoss = mlp_epoch(model, lossFunction, opt, trainLoader, argv)
        model.eval()

        # validation
        with no_grad():
            devAcc, devLoss = mlp_epoch(model, lossFunction, opt, devLoader, argv, train=False)
        print("Epoch ", i, "train acc: ", trainAcc, " train loss: ", trainLoss, "dev acc: ", devAcc, "dev loss: ",
              devLoss)

    # can evaluate on test set here


def main(argv):
    trainDirectories = getClassDirectories(argv.datadir, argv)
    X, Y = loadMIDIs(trainDirectories)
    printDatasetMetrics(X, Y)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
                                                                        test_size=0.25)  # can replace this by loading the test set instead
    doNaiveBayes(X_train, X_test, Y_train, Y_test)
    doMLP(X_train, X_test, Y_train, Y_test, argv)
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=str, help="directory to folders of midi files", required=True)
    p.add_argument("--batchsize", type=int, default=1, help="batch size")
    p.add_argument("--epochs", type=int, default=200, help="epochs for mlp")
    p.add_argument("--hidden", type=int, default=300, help="size of hidden layer in mlp")
    p.add_argument("--learning_rate", type=float, default=0.15, help="learning rate")
    p.add_argument("--dropout", type=float, default=0.00, help="probability of dropout in mlp")
    p.add_argument("--romantic", default="romantic", type=str, help="name of folder of romantic era midi files")
    p.add_argument("--baroque", default="baroque", type=str, help="name of folder of baroque era midi files")
    p.add_argument("--classical", default="classical", type=str, help="name of folder of classical era midi files")
    p.add_argument("--modern", default="modern", type=str, help="name of folder of modern/contemporary era midi files")
    arg = p.parse_args()
    main(arg)
