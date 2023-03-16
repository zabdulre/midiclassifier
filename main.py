import argparse
import os

# read the files in to midi objects
# send the object into a processing function
# the processing function creates one vector per object
# get the vector from the processing function, and put the correct label
# pass it to sci kit learn

Labels = {"romantic": [1, 0, 0, 0], "baroque": [0, 1, 0, 0], "classical": [0, 0, 1, 0], "modern": [0, 0, 0, 1]}


def getFeatures(file):
    # use music21 to open the file
    # get a vector for each feature we want to add
    # append those vectors together into one giant vector
    return


def getClassDirectories(argv):
    """
    @param argv: Contains the datadir and the name of each folder in the datadir. Each folder corresponds to an era of music.
    @return: A dictionary of directories. Each directory corresponds to a genre of music. Inside the directory should be MIDI files.
    """
    return {"romantic": argv.datadir + "/" + argv.romantic, "baroque": argv.datadir + "/" + argv.baroque,
            "classical": argv.datadir + "/" + argv.classical, "modern": argv.datadir + "/" + argv.modern}


def main(argv):
    directories = getClassDirectories(argv)

    loadedData = []  # List of examples (vectors) for each era of music, each example is a dictionary {Label : feature vector}
    for directoryEntry in directories:
        currentLabel = Labels[directoryEntry[0]]
        for fileName in os.listdir(directoryEntry[1]):
            loadedData.append({currentLabel: getFeatures(fileName)})

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=str, help="directory to folders of midi files")
    p.add_argument("--romantic", default="romantic", type=str, help="name of folder of romantic era midi files")
    p.add_argument("--baroque", default="baroque", type=str, help="name of folder of baroque era midi files")
    p.add_argument("--classical", default="classical", type=str, help="name of folder of classical era midi files")
    p.add_argument("--modern", default="modern", type=str, help="name of folder of modern/contemporary era midi files")
    arg = p.parse_args()
    main(arg)
