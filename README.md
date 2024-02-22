# Assignment 2
## CVPR 24' AI City Challenge - Detecting Violation of Helmet Rule for Motorcyclists (Motorcycle Helmet Detection)
## EECS E6691 2024 Spring

Please read the entirety of this README file, which contains description of the data/annotations, and instructions on completing the assignment. You will need to modify this file before submission.

### <span style="color:red">**[Submission Deadline]:**</span> 25th March 2024

## Introduction

The [AI City Challenge](https://www.aicitychallenge.org/), hosted at CVPR 2024, focuses on harnessing AI to enhance operational efficiency in physical settings such as retail and warehouse environments, and Intelligent Traffic Systems (ITS). **This assignment focuses on Track 5 of the AI City Challenge - Detecting Violation of Helmet Rule for Motorcyclists**. 

Motorcycles are one of the most popular modes of transportation. Due to lesser protection compared to cars and other standard vehicles, motorcycle riders are exposed to a greater risk of crashes. Therefore, wearing helmets for motorcycle riders is mandatory as per traffic rules and automatic detection of motorcyclists without helmets is one of the critical tasks to enforce strict regulatory traffic safety measures.

**In this assignment, the task is to develop and train a model to perform motorbike (and) helmet detection for riders.** The goal is to familiarize students with the end-to-end machine learning/deep learning modeling process, including data preprocessing, model selection and training, and evaluation. Students may also wish to submit their results to the evaluation server if they feel their results and methodology are competitive.

## <span style="color:red"><strong>TODO:</strong></span> Group Member Names and UNIs
**This assignment can be done in groups of upto three members** (no more than three). Please list the names and UNIs of the group members below:

- **〈Student 1 FULL NAME〉** **〈Student 1 UNI〉**
- Etc.

## <span style="color:red"><strong>TODO:</strong></span> (Re)naming of Repository

***INSTRUCTIONS*** for naming the students' solution repository for group assignments, such as the final project (students need to use a 4-letter groupID and the UNIs of all group members):
* Template: `e6691-2024spring-assign2-GroupID-UNI1-UNI2-UNI3`
* Example: `e6691-2024spring-assign2-MEME-zz9999-aa9999-aa0000`
* This can be done from the "Settings" on the repository webpage.

## Data Description

The dataset for this challenge can be downloaded from the official [AI City Challenge Tracks](https://www.aicitychallenge.org/2024-challenge-tracks/) (instructions explained below, requires sign-up).

- The training data consists of 100 videos at 10fps, each ~20 seconds long. 
- The video resolution is 1920x1080. 
- The testing data has another 100 videos, but will be released later. For this assignment, it is sufficient to use only the training data to develop/evaluate models by splitting the training data into train/val sets.

Each motorcycle in the annotated frame has bounding box annotation of each rider with or without helmet information, for upto a maximum of 4 riders in a motorcycle. The class id (labels) of the object classes in the original data is as follows:

- 1, motorbike: bounding box of motorcycle
- 2, DHelmet: bounding box of the motorcycle driver, if he/she is wearing a helmet
- 3, DNoHelmet: bounding box of the motorcycle driver, if he/she is not wearing a helmet
- 4, P1Helmet: bounding box of the passenger 1 of the motorcycle, if he/she is wearing a helmet
- 5, P1NoHelmet: bounding box of the passenger 1 of the motorcycle, if he/she is not wearing a helmet
- 6, P2Helmet: bounding box of the passenger 2 of the motorcycle, if he/she is wearing a helmet
- 7, P2NoHelmet: bounding box of the passenger 2 of the motorcycle, if he/she is not wearing a helmet
- 8, P0Helmet: bounding box of the child sitting in front of the Driver of the motorcycle, if he/she is wearing a helmet
- 9, P0NoHelmet: bounding box of the child sitting in front of the Driver of the motorcycle, if he/she is wearing not a helmet

Note: There is a .png image in the data folder that provides a visualization of the classes for better understanding.

 <span style="color:red">***IMPORTANT:***</span> For the purpose of this assignment, we will want to merge the labels into fewer classes (this needs to be done as part of the data processing step, see instructions below) in order to make the training and evaluation a little bit easier:

 - 1, Motorbike
 - 2, RiderHelmet
 - 3, RiderNoHelmet

The groundtruth file (.txt) contains bounding box information (one object instance per line) for each video. The schema is as follows (values are comma-separated):

`〈video_id〉, 〈frame〉, 〈tl_x〉, 〈tl_y〉, 〈bb_width〉, 〈bb_height〉, 〈class〉`

- 〈video_id〉 is the video numeric identifier, starting with 1. It represents the position of the video in the list of all videos, sorted in alphanumeric order.
- 〈frame〉 represents the frame count for the current frame in the current video, starting with 1.
- 〈tl_x〉 is the x-coordinate of the top left point of the bounding box.
- 〈tl_y〉 is the y-coordinate of the top left point of the bounding box.
- 〈bb_width〉 is the width of the bounding box.
- 〈bb_height〉 is the height of the bounding box.
- 〈class〉 is the class id of the object as given in the labels information above.

There is a single txt file for the training data. The testing data will be released later, **it is sufficient to split the training data into train/val sets for the purpose of model development and training.**

## <span style="color:red">**IMPORTANT:**</span> Assignment Instructions
0. Set up your computing environment (preferably GCP, since you will have to deal with a fairly large dataset, and use state-of-the-art models if you want good results). Make sure to have the NVIDIA GPU Drivers, CUDA, Jupyter, etc. installed correctly. If unsure, please refer to assignment 1 for instructions on how to set up a GCP VM instance for machine learning/deep learning development.
1. Go to the official CVPR 2024 AI City Challenge Website (https://www.aicitychallenge.org/2024-challenge-tracks/) and fill out the request form to receive access to the data. When filling out the form, make sure to select `Track 5` to get access to the Track 5 challenge data.
  <span style="color:red">***START THIS PROCESS EARLY***, it may take upto 1-2 days to receive approval and get access to the dataset</span>.
2. Download and save your copy of the dataset. You will likely need to process the raw videos and annotations into a dataset consisting of individual frame images and annotations depending on your model choice. For instance, if you are using a YOLO object detection model, you will need to process the dataset into YOLO format. 
    * You will need to split the raw data into train and val sets. <span style="color:red">***Remember to do this carefully***</span>. For example, you should select 70 videos as train and the remaining 30 videos as validation; do not include frames from the same video in both training and validation as it may lead to falsely better results. You may come up with more intelligent methods to split the data such as class-balanced splitting, etc. 
    * You may want to downsample (to a lower FPS) and select a lower resolution instead of 1920x1080 to make training faster, but note that it may affect the performance of the model.
    * <span style="color:red">***You must prepare an additional set of labels with the merged classes (described in the previous section)***</span>. This means that the annotations in your final dataset should have the three classes described above.
    * <span style="color:red">***You must include a jupyter notebook OR python script with code and comments/descriptions that demonstrates how you have processed the raw data into a formatted dataset***</span>. Save your processed dataset in google drive, and share the link with the instructors/TAs during submission.

3. Select one (or more) deep learning models to train on your dataset. You are allowed to select any model of your choice. 
    * <span style="color:red">***You must clearly reference  the literature and/or external code implementations that you have used in your code***</span>. 
    * You must use PyTorch as the framework of choice.
    * <span style="color:red">***You must include a notebook OR python script with code and comments/descriptions that demonstrates how you have trained your model***</span>.
    * There may be more elegant approaches (e.g. object detection followed by post-processing) to achieve good results on the data.

4. Evaluate your model on the dataset and analyze your plots.
    * <span style="color:red">***You must include a jupyter notebook***</span> with comments/descriptions that show training plots (loss, accuracy, mAP, etc) and per-class AP evaluation results.
    * Your evaluation should be on the merged classes described earlier. You may also evaluate on the original 9 separated classes and present the results if you wish.
    * For training plots, you are allowed to use external tools like wandb, comet, etc. for logging if you wish, but make sure to include screenshots of the plots in your notebooks.

5. Commit your changes regularly. <span style="color:red">***At least 3***</span> commits on your progress of the assignment is required before submitting a final version.

6. **Report**: Students need to submit a 1-2 page report (.pdf format only) explaining their approach, contributions by each student, data processing strategy, and discussions on results obtained along with potential improvement strategies if applicable. The report should be added to this GitHub repository.

7. **(OPTIONAL) CVPR Submission**: This step is entirely optional, student may wish to submit their results to the CVPR submission server using the following link:

    https://www.aicitychallenge.org/2024-evaluation-system/

    Note the following important points when submitting to CVPR:
    * The CVPR submission deadline is March 25th, 2024.
    * Read the Track 5 submission instructions carefully at https://www.aicitychallenge.org/2024-data-and-evaluation/.
    * For this assignment, we evaluate on the smaller merged class labels; official submission requires you to submit results based on the original 9-class labels.
    * Your results will need to be competitive :)
    * Refer to the instructions page on the exact submission format.

## Submission

### <span style="color:red">**[Submission Deadline]:**</span> 25th March 2024

1. Include all your code in the GitHub repository before submission. **Do not include any data or labels in the GitHub**, instead you must put your processed dataset in a google drive folder, and share the link the in description (TODO by students below).

2. Similarly, include model weights in the google drive folder.

3. Add your report (.pdf file) to the GitHub repository before submission.

4. Submission will be through Gradescope. Students will need to enter gradescope and link the GitHub repo for submission. For instructions on group submission in Gradescope, please refer to [this link](https://hmc-cs-131-spring2020.github.io/howtos/assignments.html).

## <span style="color:red"><strong>TODO:</strong></span> Repository Description

This section must be filled up by students with a brief description of the approach, organization of code in the directories, and instructions on running python scripts/models (if applicable).

### Dataset/Model Google Drive Link:
TODO add your group's dataset/model google drive folder link (make sure to set the permissions accordingly before sharing the link).

## Grading

- Data Preprocessing/Dataset preparation: <span style="color:red">(30%)</span>
- Model Selection/Training <span style="color:red">(30%)</span>
- Evaluation and Results <span style="color:red">(20%)</span>
- Report <span style="color:red">(10%)</span>
- GitHub Code and Organization of Repository <span style="color:red">(10%)</span>
