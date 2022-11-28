import matplotlib
# matplotlib.interactive(False)
import os
import warnings

warnings.filterwarnings("ignore")
from statistics import mean, median

import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as no
import sklearn
from mutagen.mp3 import MP3
from tqdm import tqdm

birds = []
DATASET_URL = '/media/sasanka/Expansion/xeno-canto-bird-recordings-extended-a-m/A-M/'
for root, dirs, files in os.walk(DATASET_URL):
    if root == DATASET_URL:
        birds = dirs
birds50 = []
flist = []
blist = []
i50 = 0
for i, bird in enumerate(birds):
    for root, dirs, files in os.walk(DATASET_URL + bird):
        for file in files:
            if file.endswith(".mp3"):
                blist.append(os.path.join(root, file))
    if len(blist) > 50:
        i50 = i50 + 1
        birds50.append(bird)
        flist.append(blist)
    blist = []

# print(len(birds))

def saveMel(y, directory):
    N_FFT = 1024  # Number of frequency bins for Fast Fourier Transform
    HOP_SIZE = 1024  # Number of audio frames between STFT columns
    SR = 44100  # Sampling frequency
    N_MELS = 30  # Mel band parameters
    WIN_SIZE = 1024  # number of samples in each STFT window
    WINDOW_TYPE = "hann"  # the windowin function
    FEATURE = "mel"  # feature representation

    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(6, 6)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(
        librosa.core.amplitude_to_db(
            librosa.feature.mfcc(
                dct_type=3,
                y=y,
                sr=SR,
                n_fft=N_FFT,
                hop_length=HOP_SIZE,
                n_mels=N_MELS,
                htk=True,
                fmin=0.0,
                fmax=SR / 2.0,
            ),
            ref=1.0,
        ),
        sr=SR,
        hop_length=HOP_SIZE,
    )
    fig.savefig(directory)
    fig.clear()
    ax.cla()
    plt.clf()
    plt.close("all")

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def saveMel2(y, directory):

    N_FFT = 1024  # Number of frequency bins for Fast Fourier Transform
    HOP_SIZE = 1024  # Number of audio frames between STFT columns
    SR = 44100  # Sampling frequency
    N_MELS = 30  # Mel band parameters
    WIN_SIZE = 1024  # number of samples in each STFT window
    WINDOW_TYPE = "hann"  # the windowin function
    FEATURE = "mel"  # feature representation

    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    plt.rcParams["figure.figsize"] = (10, 2)
    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(4, 4)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax1 = plt.subplot(2, 1, 1)
    spectogram = librosa.display.specshow(
        librosa.core.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=SR)),
        x_axis="time",
        y_axis="mel",
    )
    plt.subplot(2, 1, 2, sharex=ax1)
    librosa.display.waveshow(y=y, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color="r")

    fig.savefig(directory)
    fig.clear()
    # ax.cla()
    plt.clf()
    plt.close("all")

size = {
    "desired": 10,  # [seconds]
    "minimum": 5,  # [seconds]
    "stride": 0,  # [seconds]
    "name": 5,
}
step = 1
if step > 0:
    for bird, birdList in enumerate(flist):
        for birdnr, path in tqdm(enumerate(birdList)):
            directory = (
                "./data/mels-19class2/" + str(bird) + birds50[bird][: size["name"]] + "/"
            )
            if not os.path.exists(directory):
                os.makedirs(directory)
            if not os.path.exists(
                directory + path.rsplit("/", 1)[1].replace(" ", "")[:-4] + "1_1.png"
            ):
                y, sr = librosa.load(path, mono=True)
                y = no.reduce_noise(y, y_noise=y, sr=44100)
                step = (size["desired"] - size["stride"]) * sr
                nr = 0
                for start, end in zip(
                    range(0, len(y), step), range(size["desired"] * sr, len(y), step)
                ):
                    nr = nr + 1
                    if end - start > size["minimum"] * sr:
                        melpath = path.rsplit("/", 1)[1]
                        melpath = (
                            directory
                            + melpath.replace(" ", "")[:-4]
                            + str(nr)
                            + "_"
                            + str(nr)
                            + ".png"
                        )
                        saveMel2(y[start:end], melpath)
            pass
else:
    print("Error: Stride should be lower than desired length.")