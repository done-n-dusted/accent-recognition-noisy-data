# Impact of Noise for Accent Recognition
## Term Project
#### CSE583 - Pattern Recognition and Machine Learning
Anurag Pendyala

> Refer to `Report.pdf` to find more details regarding the project.

## Requirements

All the packages required for the project can be found in `requrements.txt` file. All the installations are straight forward except for Augly. You can follow FAIR's [link](https://github.com/facebookresearch/AugLy) to follow their instructions for installations.

## Adding Noise for Augmentation
`creating_noisy_audio.py` contains a function that takes a path for an audio file as an input and adds a noise to it. The locaiton for the background noise you wish to add must also be passed as an argument. The notebook `create_noise_dataset.ipynb` demonstrates how to add noise to a batch of files from `csv` files described in the **Data Format** Section.

## Data Format

You can download the data from the [link](https://pennstateoffice365-my.sharepoint.com/:f:/r/personal/app5997_psu_edu/Documents/PRML%20Data/data_for_term?csf=1&web=1&e=KAqwVX) provided in the data folder. It is a subset of AccentDB data set on which noise augmentation was performed using Augly. The provided data folder contains a lot of `csv` and `npy` files as well which will be explained soon.

Every set of data should have a corresponding `csv` file with two columns `location` and `label` corresponding to location of the audio file and label, in this case the accent class of that audio file. The location is the path to the audio file. For example, lets take the `no_noise_train.csv` file. This is a dataset that contains location to all clean audio files needed to train the model. 

Once the data is arranged in such a way, you can run the function in `creating_noisy_audio.py` file to generate the noisy version for the particular audio file. Once the noisy files are generated, they have to organized similar to that of `no_noise_train.csv` with any name of your choice. 

With all the data directories and appropriate `csv` files, your data is ready. However, they are still audio files, and the model requires MFCCs extracted from each of these audio files as features.

## Formating the data for the Model

The `csv` files contains the location and class of the audio file. The model, however, requires 13 MFCC features from these individual audio files. The file `extract_mfccs.py` can be used to do that task Run the following command to run to create a `npy` file from the `csv` file which can be fed into the model. The `npy` contains MFCC features for every audio file along with its label.

```
python extract_mfccs.py path/to/csv path/to/save_npy
```

Create and format as many datasets as needed for you to train and test the model. Save them in a folder.

## Training the model

> `models.py` contains the ConvNet used for the project. You can add new networks and train and test the data.

The file `train.py` code trains and tests the model on given datasets. It also needs a model name to later save the model in `models/<model_name>.pth` along with history corresponding to training, validation losses and accuracis in npy files in `history/<model_name>.npy`. Run the following command to train. Note that, by default, the model trains on the available GPU.

```
python train.py --train path/to/train1.npy path/to/train2.npy --test path/to/test1.npy path/to/test2.npy ... --num_epochs 10 --model model_name

```

Once the `--train` or `--test` arguments start, you can give locations for as many dataset .npy files as needed. The code will eventually combine all of them to train and test the model on them. `--num-epochs` argument is not compulsory and is `10` by default.

## Testing the Model

> All saved models can be found in `models/` directory.

Similar to `train.py` run the following command to test the model.

```        
python test.py --model path/to/model --test path/to/test1.npy path/to/test2.npy ...
```

`--model` argument takes the model path to be tested. You can add locations to any number of datasets after the `--test` argument and everything will be eventually combined. 

The file outputs overall and classwise accuracies for the considered classes. 

## Description for other files

* `utils.py` contains important utility functions that are necessary. One of the important functions is the function used to save mfcc spectrum as a png from a given audio file.
* `history/` directory contains `.npy` files with information corresponding to training of that particular model.
* `noises/` directory contains background noises that I have considered for the task. They are listed below:
    * Cafe
    * Babble
    * White Static

    These were taken from [Pixabay](https://pixabay.com/)