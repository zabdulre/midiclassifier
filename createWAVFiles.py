import argparse
import os
import music21

def getClassDirectories(rootdir, argv):
    return {"romantic": rootdir + "/" + argv.romantic, "baroque": rootdir + "/" + argv.baroque,
        "classical": rootdir + "/" + argv.classical, "modern": rootdir + "/" + argv.modern}

def createWavFromMidi(filePath: str):
        try:
            midiObject = music21.converter.parse(filePath)
        except:
            print(filePath, "could not be parsed, skipping...")
            return

        wavFilePath = os.path.splitext(filePath)[0] + ".wav"

        #TODO
        #create wav from midi object here

def createFiles(directories):
    for folder in directories.items():
        # Debug: only run on one folder
        #if (folder[0] != "modern"):
        #    continue

        for fileName in os.listdir(folder[1]):
            if fileName[0] == '.':  # skip any hidden files
                continue
            currentFileDirectory = folder[1] + "/" + fileName
            createWavFromMidi(currentFileDirectory)

    return

def main(argv):
    trainDirectories = getClassDirectories(argv.datadir, argv)
    testDirectories = getClassDirectories(argv.testdir, argv)
    createFiles(trainDirectories)
    createFiles(testDirectories)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=str, help="directory to folders of midi files", required=True)
    p.add_argument("--testdir", type=str, help="directory to folders of midi files", default=None)
    p.add_argument("--romantic", default="romantic", type=str, help="name of folder of romantic era midi files")
    p.add_argument("--baroque", default="baroque", type=str, help="name of folder of baroque era midi files")
    p.add_argument("--classical", default="classical", type=str, help="name of folder of classical era midi files")
    p.add_argument("--modern", default="modern", type=str, help="name of folder of modern/contemporary era midi files")
    arg = p.parse_args()
    main(arg)