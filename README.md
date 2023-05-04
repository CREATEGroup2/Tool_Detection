Project Title

This repository contains the code and necessary instructions to participate in the challenge. Please follow the steps below to download the data, prepare the dataset for training, and train the networks.
Download Data

The download links for the data are password protected and were available until May 5th, 2023. Registered participants received the password via email on April 24th, 2022.
Training Data:

The training data can be downloaded in 8 parts using this repo: https://github.com/CREATE-Table-Top-Challenge/Central_Line_Challenge/tree/main

Unlabelled Data:

The unlabelled data can be found here. Participants may upload labels using the following form until 12:00pm EST (noon) May 4th, 2023. As submissions pass through the review process, they will be made available here.
Test Data:

The test data can be downloaded using the following link on May 4th, 2023.
Prepare Dataset for Training

After downloading all parts of the dataset for training, clone this repository. Navigate to the location where the code is located and use the prepareDataset.py script to unpack and format your dataset. The script can be run by entering the following lines into your command prompt (replace all instances of UserName with your real username):

    conda activate createKerasEnv
    cd <path_to_repository>
    python prepareDataset.py --compressed_location=C:/Users/UserName/Downloads --target_location=C:/Users/UserName/Documents/CreateChallenge --dataset_type=Train

To prepare the test set, follow the same steps, but change the --dataset_type flag to Test. The process is the same for the unlabelled data except --dataset_type should be "Unlabelled".

If the code is executed correctly, you should see a new directory in your target location called either Training_Data or Test_Data. These directories will contain a set of subdirectories (one for each video) that contain the images. Within the main folder, you will also see a csv file that contains a compiled list of all images and labels within the dataset. (Note: there will not be any labels for the test images).
Training the networks

Begin by activating your conda environment:

    conda activate createKerasEnv

Next, select which network you would like to run.

One baseline network has been provided for each subtask of the challenge:
Subtask 1: Surgical tool localization/ detection

    Baseline network folder: Tool Detection

    Model: Yolo-v3

    Inputs: single image

    Outputs: list of dictionaries with the form {'class': classname, 'xmin': int, 'xmax': int, 'ymin': int, 'ymax': int, 'conf': float}
        Download the backend weights: Backend weights
        Place weights in the Tool_Detection directory
        Train the network (replace paths as necessary):

    python your_path/Tool_Detection/Train_Yolov3.py --save_location=C:/Users/SampleUser/Documents/toolDetectionRun1 --data_csv_file=C:/Users/SampleUser/Documents/Training_Data/Training_Data.csv


Required flags:

    --save_location: The folder where the trained model and all training information will be saved
    --data_csv_file: File containing all files and labels to be used for training
