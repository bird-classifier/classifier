import os

# define number of files for all sets
train = 0.8 # 80% of all sound should be in the train set
val = 0.1 # 10% validation set
test = 0.1 # 10% test set
kfolds = 1
basePath='/media/sasanka/Expansion/xeno-canto-bird-recordings-extended-a-m/A-M/' # path with sound files. Can be downloaded with "AM_downloadData"
imPath = "./data/mels-19class2/" # path with images (melspectrograms)
                                # Can be generated with "AM_prepareData" after downloading sound files
destPath = "./data/19_class2/" # destination path - where the split dataset should be copied
                            # This folder will be used to train CNNs

# first find all of the mp3 files in the directory
birds=[] # list of all bird spiecies (Ember, Phyll...)
singleBirdList=[] # list of files for one bird
allFilesList=[] # list of all files for all birds. A list of singleBirdLists. 
for root, dirs, files in os.walk(basePath):
    if root == basePath:
        birds=dirs
birds = ['amecro', 'amegfi', 'amered', 'aldfly']
trainSet=[]
testSet=[]
valSet=[]

birdsShort=[] # list of short file names
birdNumber=0
for nr,bird in enumerate(birds):
    for root, dirs, files in os.walk(basePath+bird):
        for file in files:
            if file.endswith(".mp3"):
                singleBirdList.append(os.path.join(root, file))
    print(bird, len(singleBirdList))
    if len(singleBirdList) > 50:
        birdsShort.append(str(birdNumber)+bird[:5])
        birdNumber = birdNumber+1
        print("Found ", len(singleBirdList), ' mp3 files for ', bird)
        trainSet.append(int(train*len(singleBirdList)))
        valSet.append(int(val*len(singleBirdList)))
        roundDiff = len(singleBirdList)-(int(train*len(singleBirdList))+int(test*len(singleBirdList))+int(val*len(singleBirdList))) 
        testSet.append(int(test*len(singleBirdList)) + roundDiff)
        print("Size of train: ", int(train*len(singleBirdList)),", val: ",int(val*len(singleBirdList)) ,", test: ",int(test*len(singleBirdList)))
        allFilesList.append(singleBirdList)
    singleBirdList=[]


# print(trainSet)
# print(valSet)
# print(testSet)

from random import sample

trainFiles=[]
valFiles=[]
testFiles=[]

for index, singleBirdList in enumerate(allFilesList):
    randFiles = sample(range(len(singleBirdList)), len(singleBirdList))
    start = 0
    end = trainSet[index]
    trainFiles.append(randFiles[start:end])
    start = end 
    end = start + valSet[index]
    valFiles.append(randFiles[start:end])
    start = end 
    end = start + testSet[index] 
    testFiles.append(randFiles[start:end])
    print("Selected random files number:\n train: ", len(trainFiles[index]),"/",trainSet[index],
          ", val: ",len(valFiles[index]),"/",valSet[index],
          ", test: ",len(testFiles[index]),"/",testSet[index])

def extractName(string):
    return string.rsplit('/',1)[1].replace(' ', '')[:-4]

# sort all the lists to make copying files easier
sets=[trainFiles,valFiles,testFiles]
for fileSet in sets:
    for index,files in enumerate(fileSet):
        fileSet[index].sort()


# change full names to short

for root, dirs, files in os.walk(basePath):
    if root == basePath:
        birds=dirs
#birdsShort=[]
#for bird in birds:
#    birdsShort.append(bird[:5])

setNames = ["train/","val/","test/"]

print("Long: ", birds,"\nShort: ",birdsShort)

import shutil

counter=0
for birdNumber, bird in enumerate(birdsShort): # for each class (bird) check where the file should be copied
    print(counter)
    counter=0
    for setName, fileSet in zip(setNames, sets): # check for all datasets: train, val and test sests 
        for setNumber in fileSet[birdNumber]:
            
            for fileNumber, file in enumerate(allFilesList[birdNumber]):
                if setNumber == fileNumber: # if file number to copy is same as number of file, then copy it
                    
                    for root, dirs, files in os.walk(imPath):
                        for file2 in files:
                            if extractName(file) in file2:
                                counter=counter+1
                                source=root+"/"+file2
                                
                                destination = destPath+setName+bird+"/"
                                if not os.path.exists(destination):
                                    os.makedirs(destination)
                                shutil.copy2(source, destination)
                                #print(source, "   ->   ", destination)