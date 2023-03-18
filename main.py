import argparse
import os
import music21

# read the files in to midi objects
# send the object into a processing function
# the processing function creates one vector per object
# get the vector from the processing function, and put the correct label
# pass it to sci kit learn

Labels = {"romantic": 1, "baroque": 2, "classical": 3, "modern": 4}
loadedData = []  # List of examples (vectors) for each era of music, each example is a dictionary {Label : feature vector}
labelsToLoad = []
filesToLoad = []

count=0

#List of all key signatures
ksList = []
for a in ["C","D","E", "F", "G", "A", "B"]:
    for b in ["","-"]:
        for c in [" major", " minor"]:
            ksList.append(a+b+c)

#TODO
def getFeaturesFromMIDIObject(midiObject):
    features = []

    #Get Enumerated Key Signature
    try:
        ks = midiObject.flat.keySignature.asKey()
        ksIndex = ksList.index(str(ks))
        print(ks, ksIndex)
        features.append(ksIndex)
    except Exception as e:
        print(e)
        global count
        count +=1
        print("does not have keysig? Total bad files:", count)
        features.append(-1)

    #Get 

    return features

def updateLoadedData(pos, length, out, result):
    if out is not None:
        print("Parsed file number ", pos + 1, " ", result)
        #if len(labelsToLoad) > pos: #only add to the back of the list, prevents duplicates
        if pos == len(loadedData):
            loadedData.append([labelsToLoad[pos], getFeaturesFromMIDIObject(out)])
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
    global count
    print("total bad files: ", count)
    print("Done!")
    return loadedData


def getClassDirectories(argv):
    """
    @param argv: Contains the datadir and the name of each folder in the datadir. Each folder corresponds to an era of music.
    @return: A dictionary of directories. Each directory corresponds to a genre of music. Inside the directory should be MIDI files.
    """
    return {"romantic": argv.datadir + "/" + argv.romantic, "baroque": argv.datadir + "/" + argv.baroque,
            "classical": argv.datadir + "/" + argv.classical, "modern": argv.datadir + "/" + argv.modern}


def main(argv):
    directories = getClassDirectories(argv)


    for folder in directories.items():

        #Debug: only run on one folder
        #if(folder[0]!="classical"):
        #    continue

        currentLabel = Labels[folder[0]]
        for fileName in os.listdir(folder[1]):
            if fileName[0] == '.':  # skip any hidden files
                continue
            currentFileDirectory = folder[1] + "/" + fileName
            filesToLoad.append(currentFileDirectory)
            labelsToLoad.append(currentLabel)
            #currentFeatures = getFeatures(currentFileDirectory)
            #loadedData.append({currentLabel: currentFeatures})

    loadedData = getFeatures()

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
