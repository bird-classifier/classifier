# Bird Species Identification from Audio Data

Dataset used: https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m

Report: [Link](https://github.com/bird-song-classifier/bird-classifier/blob/7bcbb95c6488bcc9c753b8df2328c2b5ab041642/report.pdf)

## File structure
```
.
├── demo: Files and notebooks used for demo
├── extracted_data.csv: CSV file containing the features extracted from audio files
├── feature_extractor.ipynb: Notebook to extract features from audio
├── mfcc_models_cross.ipynb: Notebook to run models with cross validation
├── mfcc_models.ipynb: Notebook to run models(witout cross validation) 
├── models: Folder with saved models without cross validation
├── models_cross: Folder with saved models with cross validation
├── Pipfile
├── Pipfile.lock
├── plots: Folder with saved plots
├── README.md
├── testing_preprocessing.ipynb: Notebook where we experimented preprocessing methods
└── train_extended.csv: Metadata of the dataset
```

## Steps to setup the project
1. Clone the repository from https://github.com/bird-song-classifier/classifier
2. Create a python virtual environment.
    ```
    $ pipenv shell
    ```
3. Install the packages.
    ```
    $ pipenv install
    ```
4. For feature extraction, run the ```feature_extractor.ipynb```.
5. To train the models, run ```mfcc_models.ipynb``` and ```mfcc_models_cross.ipynb```.
