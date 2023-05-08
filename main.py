import argparse
import os
import music21
import torch
from sklearn import model_selection, naive_bayes, metrics, preprocessing
from collections import Counter
import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
from music21 import analysis
from torch import tensor, nn, optim, no_grad, argmax, sum, flatten
from torch.utils.data import DataLoader, dataset
from music21 import pitch
from Model import MLPDoubleModel, MLPTripleModel, CNNModel
from tqdm import tqdm
import librosa

# read the files in to midi objects
# send the object into a processing function
# the processing function creates one vector per object
# get the vector from the processing function, and put the correct label
# pass it to sci kit learn

timidityPath = '/usr/local/bin/timidity'
Labels = {"romantic": 0, "modern": 1, "classical": 2, "baroque": 3}
Numbers = {0: "romantic", 1: "modern",2: "classical", 3:"baroque"}









loadedData = []  # List of examples (vectors) for each era of music, each example is a dictionary {Label : feature vector}
loadedLabels = []
loadedFiles = [] #files in order of labels
loadedObjects = []
loadedIndices = []
labelsToLoad = []
filesToLoad = []
nbPredictions = []
mlpPredictions = []
cnnPredictions = []
CNNFiles = []
# Number of valid files that had errors when analyzing/extracting features (note: does not account for files that were unable to be parsed in the first place)
badfilecount = 0
numF = 0
cnnFTMatrixDimensions = (1,1) #TODO
# List of all key signatures
# ksList = []
# for a in ["c", "d", "e", "f", "g", "a", "b"]:
#     for b in ["", "#", "-"]:
#         for c in [" major", " minor"]:
#             ksList.append(a + b + c)


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
            loadedFiles.append(filesToLoad[pos])
            loadedObjects.append(out)
            loadedIndices.append(len(loadedIndices))
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
    loadedFiles.clear()
    loadedIndices.clear()
    loadedObjects.clear()

    for folder in directories.items():

        # Debug: only run on one folder
        # if (folder[0] != "modern"):
        #     continue

        currentLabel = Labels[folder[0]]
        for fileName in os.listdir(folder[1])[:10]:
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
        aa.pie(Counter(Y).values(), labels=[Numbers[i] for i in list(Counter(Y).keys())],
               autopct='%1.0f%%')  # change the labels according to if the dataset has all eras or not TODO
        plt.show()
        pass  # print label related metrics here
    if X is not None and Y is not None:
        pass  # print metrics related to both here


def doNaiveBayes(X_train, X_dev, X_test, Y_train, Y_dev, Y_test):
    nb = naive_bayes.GaussianNB()
    nb.fit(X_train, Y_train)
    print("Naive Bayes results on dev set: ")
    print(metrics.classification_report(Y_dev, nb.predict(X_dev)))
    if X_test is not None and Y_test is not None:
        print("Naive Bayes results on test set: ")
        print(metrics.classification_report(Y_test, nb.predict(X_test)))
        global nbPredictions
        nbPredictions = nb.predict(X_test).tolist()


def mlp_loaders(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, argv):
    trainDataset = dataset.TensorDataset(tensor(X_train,dtype=torch.float32), tensor(Y_train,dtype=torch.float32))
    devDataset = dataset.TensorDataset(tensor(X_dev,dtype=torch.float32), tensor(Y_dev,dtype=torch.float32))
    if (X_test is None) or (Y_test is None):
        return DataLoader(trainDataset, argv.batchsize, shuffle=True), DataLoader(devDataset, argv.batchsize,
                                                                               shuffle=False), None
    else:
        testDataset = dataset.TensorDataset(tensor(X_test, dtype=torch.float32), tensor(Y_test, dtype=torch.float32))
        return DataLoader(trainDataset, argv.batchsize, shuffle=True), DataLoader(devDataset, argv.batchsize,
                                                                               shuffle=True), DataLoader(testDataset, len(testDataset), shuffle=False)



#For each wav file, get Short-Time Fourier Transform.
#We then take the magnitude of the spectrum. Getting us a frequency x timestep matrix.
def wavToCNNInput(Wav_filepath, max_time=512, max_freq=512, maxLength=100000):
    audio, sample_rate = librosa.load(Wav_filepath)

    iterations = len(audio) // maxLength
    remainingSize = len(audio)
    clips = []
    for i in range(max(iterations,0)):
        item = np.abs(librosa.stft(audio[i*maxLength:((i+1)*maxLength)], hop_length=500, n_fft=400))
        item = preprocessing.StandardScaler().fit_transform(item)
        clips.append(item)
        remainingSize -= maxLength

    if remainingSize > 0:
        item = np.abs(librosa.stft(np.pad(audio[iterations*maxLength:], (maxLength-remainingSize), mode="constant")[:maxLength], hop_length=500, n_fft=400))
        item = preprocessing.StandardScaler().fit_transform(item)
        clips.append(item)
    out = clips.copy()
    '''
    if audio.shape[0] < maxLength:
        audio = np.pad(audio, (maxLength - audio.shape[0]), mode="constant")
        audio = audio[:maxLength]
    else:
        audio = audio[:maxLength]
    spectrum = librosa.stft(audio, hop_length=40000, n_fft=200) #TODO make this much smaller
    out = np.abs(spectrum)
    out = preprocessing.StandardScaler().fit_transform(out)
    #2048, 1025 were og values
    '''
    '''
    if out.shape[1] < max_time: #TODO make this much smaller
        out = np.pad(out, (0, max_time - out.shape[1]), constant_values=0.0)
    else:
        out = out[:,:max_time]

    if out.shape[0] < max_freq: #TODO make this much smaller
        out = np.pad(out, (max_freq - out.shape[0], 0), constant_values=0.0)
        out = out[:,:max_time]
    else:
        out = out[:max_freq,:]
    '''

    return out

#For each file, get & add the FT Matrix (input to CNN)
#Run the list of FTMatrices (so tensor is 3D)
def getMusicFTMat(batchOfFileIndices, listOfFiles):
    listOfFTMatrices = []
    for index in tqdm(batchOfFileIndices):
        midiFileName = listOfFiles[int(index.item())]
        #rawWav = getRawWav(midiFileName) #can pass in midi file name or midi object here
        listOfFTMatrices.extend(wavToCNNInput(midiFileName))

    #return torch.tensor(listOfMusicFeatures)
    return torch.tensor(listOfFTMatrices)


def mlp_epoch(model, lossFunction, opt, loader, argv, train=True, test=False):
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

        if test:
            global mlpPredictions
            mlpPredictions.append(flatten(predictions).tolist())

    return numCorrect / numExamples, totalLoss



def doMLP(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, argv, isCNN=False):
    trainLoader, devLoader, testLoader = mlp_loaders(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, argv)

    if isCNN:
        model = CNNModel(len(Labels.keys()), cnnFTMatrixDimensions, argv)
    else:
        if argv.two_layer:
            model = MLPDoubleModel(len(Labels.keys()), numF, argv)
        else:
            model = MLPTripleModel(len(Labels.keys()), numF, argv)

    lossFunction = nn.CrossEntropyLoss()
    if argv.sgd:
        opt = optim.SGD(model.parameters(), argv.learning_rate)
    else:
        opt = optim.Adam(model.parameters(), argv.learning_rate)

    trainAccList = []
    trainLossList = []
    devAccList = []
    devLossList = []
    for i in tqdm(range(argv.epochs)):
        model.train()
        trainAcc, trainLoss = mlp_epoch(model, lossFunction, opt, trainLoader, argv)
        model.eval()

        #validation
        with no_grad():
            devAcc, devLoss = mlp_epoch(model, lossFunction, opt, devLoader, argv, train=False)
        print("Epoch ", i, "train acc: ", trainAcc, " train loss: ", trainLoss, "dev acc: ", devAcc, "dev loss: ",
              devLoss)

        trainAccList.append(trainAcc)
        trainLossList.append(trainLoss)
        devAccList.append(devAcc)
        devLossList.append(devLoss)

    if testLoader is not None:
        with no_grad():
            testAcc, testLoss = mlp_epoch(model, lossFunction, opt, testLoader, argv, train=False, test=True)
        print("Test acc: ", testAcc, " test loss: ", testLoss)

    toPrint = ""
    if isCNN:
        toPrint = "CNN: "
    toPrint += str(argv.learning_rate) + ", hidden: "+ str(argv.hidden) + ", batch: " + str(argv.batchsize) + ", dropout: " + str(argv.dropout) + ", sgd: " + str(argv.sgd)

    plt.title("Train acc " + toPrint)
    plt.plot([i for i in range(argv.epochs)], trainAccList)
    plt.xlabel("epoch")
    plt.ylabel("Training accuracy")
    plt.show()

    plt.title("Train loss " + toPrint)
    plt.plot([i for i in range(argv.epochs)], trainLossList)
    plt.xlabel("epoch")
    plt.ylabel("Training loss")
    plt.show()

    plt.title("Dev accuracy " + toPrint)
    plt.plot([i for i in range(argv.epochs)], devAccList)
    plt.xlabel("epoch")
    plt.ylabel("Dev accuracy")
    plt.show()

    plt.title("Dev loss  " + toPrint)
    plt.plot([i for i in range(argv.epochs)], devLossList)
    plt.xlabel("epoch")
    plt.ylabel("Dev loss")
    plt.show()

    # can evaluate on test set here


def loadWAVs(directories):
    X = []
    Y = []
    CNNFiles.clear()
    for folder in directories.items():
        # Debug: only run on one folder
        #if (folder[0] != "modern"):
        #    continue

        currentLabel = Labels[folder[0]]
        for fileName in os.listdir(folder[1]):
            if fileName[0] == '.':  # skip any hidden files
                continue
            currentFilePath = folder[1] + "/" + fileName
            print("Parsing " + currentFilePath)
            #CNN_temp_files.append(currentFileDirectory)
            #indices.append(len(indices))
            clips = wavToCNNInput(currentFilePath)
            X.extend(clips)
            [Y.append(currentLabel) for i in clips]
            [CNNFiles.append(currentFilePath) for i in clips]

    return np.stack(X, 0) , Y


def main(argv, X=None, Y=None, X_filenames=None):
    '''
    if (X is None) or (Y is None):
        trainDirectories = getClassDirectories(argv.datadir, argv)
        X, Y = loadMIDIs(trainDirectories)

    if argv.testdir is None:
        X_test = None
        Y_test = None
    else:
        testDirectories = getClassDirectories(argv.testdir, argv)
        X_test, Y_test = loadMIDIs(testDirectories)
        printDatasetMetrics(X_test, Y_test, "Test dataset")

    printDatasetMetrics(X, Y)
    X_train, X_dev, Y_train, Y_dev = model_selection.train_test_split(X, Y,
                                                                      test_size=0.25)  # can replace this by loading the test set instead
    doNaiveBayes(X_train, X_dev, X_test, Y_train, Y_dev, Y_test)
    doMLP(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, argv)

    if argv.testdir is not None:
        for i in range(len(Y_test)):
            print("For ", loadedFiles[i], " nb predicted: ", Numbers[nbPredictions[i]], ", mlp predicted: ",
                  Numbers[mlpPredictions[0][i]])
    '''
    if argv.wavdatadir is not None:
        wavTrainDirs = getClassDirectories(argv.wavdatadir, argv)
        X, Y = loadWAVs(wavTrainDirs)
        X_train, X_dev, Y_train, Y_dev = model_selection.train_test_split(X, Y,
                                                                          test_size=0.25)
        if argv.wavtestdir is None:
            X_test = None
            Y_test = None
        else:
            wavTrainDirs = getClassDirectories(argv.wavtestdir, argv)
            X_test, Y_test = loadWAVs(wavTrainDirs)
        mlpPredictions.clear()
        doMLP(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, argv, True)

    #TODO make this include CNN
    if argv.wavtestdir is not None:
        for i in range(len(Y_test)):
            print("For ", CNNFiles[i], " cnn predicted: ", Numbers[mlpPredictions[0][i]])

'''
def scripts(p):

    script1 = "--datadir dataset --batchsize 10 --hidden 200 --learning_rate 0.0055 --epochs 100 --dropout 0.03 --sgd".split()
    script2 = "--datadir dataset --batchsize 1 --hidden 200 --learning_rate 0.015 --epochs 100 --dropout 0.03".split()
    script3="--datadir dataset --batchsize 10 --hidden 200 --learning_rate 0.0055 --epochs 100 --dropout 0.0".split()
    script4="--datadir dataset --batchsize 50 --hidden 300 --learning_rate 0.0035 --epochs 300 --dropout 0.03 ".split()
    script5="--datadir dataset --batchsize 50 --hidden 300 --learning_rate 0.0035 --epochs 300 --dropout 0.03".split()
    script6="--datadir dataset --batchsize 100 --hidden 300 --learning_rate 0.0025 --epochs 700 --dropout 0.03 --testdir testset".split()
    argv = p.parse_args(script1)
    trainDirectories = getClassDirectories("dataset", argv)
    X, Y = loadMIDIs(trainDirectories)
    print("--------Script1--------------")
    main(p.parse_args(script1), X, Y)
    print("--------Script2--------------")
    main(p.parse_args(script2), X, Y)
    print("--------Script3--------------")
    main(p.parse_args(script3), X, Y)
    print("--------Script4--------------")
    main(p.parse_args(script4), X, Y)
    print("--------Script5--------------")
    main(p.parse_args(script5), X, Y)
    print("--------Script6--------------")
    main(p.parse_args(script6), X, Y)
    return
'''

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=str, help="directory to folders of midi files", required=True)
    p.add_argument("--testdir", type=str, help="directory to folders of midi files", default=None)
    p.add_argument("--wavdatadir", type=str, help="directory to folders of wav files", default=None)
    p.add_argument("--wavtestdir", type=str, help="directory to folders of wav files", default=None)
    p.add_argument("--batchsize", type=int, default=100, help="batch size")
    p.add_argument("--epochs", type=int, default=5, help="epochs for mlp")
    p.add_argument("--hidden", type=int, default=300, help="size of hidden layer in mlp")
    p.add_argument("--learning_rate", type=float, default=0.0025, help="learning rate")
    p.add_argument("--sgd", action='store_true', help="Whether to use sgd for optimizer. Default is adam")
    p.add_argument("--dropout", type=float, default=0.03, help="probability of dropout in mlp")
    p.add_argument("--two_layer", action='store_true', help="Use a 2 layered mlp, instead of 3 layered")
    p.add_argument("--romantic", default="romantic", type=str, help="name of folder of romantic era midi files")
    p.add_argument("--baroque", default="baroque", type=str, help="name of folder of baroque era midi files")
    p.add_argument("--classical", default="classical", type=str, help="name of folder of classical era midi files")
    p.add_argument("--modern", default="modern", type=str, help="name of folder of modern/contemporary era midi files")
    #comment these two lines out to run scripts (Don't have to reload train data each time)
    arg = p.parse_args()
    main(arg)
    #scripts(p)

