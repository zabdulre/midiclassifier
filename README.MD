The steps to run this code are

1) Install Timidity to generate the wav files. You can install timidity by doing, brew install timidity

2) If you do not already have the WAV files, you may generate them by using the script in the midi directory. 

Simply create a wav folder in the base midiclassifier directory. In this wav folder, create a "dataset" and "testset" folder. In each of these two folders, create a "Modern" "Romantic" "Baroque" and "Classical" folder. Then, run the createWAVFiles script with the --datadir and --testdir arguments pointing to the newly created dataset and testset folder.

3) Install the requirements in requirements.txt 

4) You can run all pipelines by simply running main.py without any arguments. 
If you would like to use arguments, please modify main.py by commenting out line 683 and uncommenting lines 681 and 682. Please refer to this section of the code to see which arguments are available. You may also refer to the example commands given in experimentScripts.txt.

Thank you!
