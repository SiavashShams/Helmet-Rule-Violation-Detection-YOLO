## CVPR 24' AI City Challenge - Detecting Violation of Helmet Rule for Motorcyclists (Motorcycle Helmet Detection)


Motorcycles are one of the most popular modes of transportation. Due to lesser protection compared to cars and other standard vehicles, motorcycle riders are exposed to a greater risk of crashes. Therefore, wearing helmets for motorcycle riders is mandatory as per traffic rules and automatic detection of motorcyclists without helmets is one of the critical tasks to enforce strict regulatory traffic safety measures.


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


The groundtruth file (.txt) contains bounding box information (one object instance per line) for each video. The schema is as follows (values are comma-separated):

`〈video_id〉, 〈frame〉, 〈tl_x〉, 〈tl_y〉, 〈bb_width〉, 〈bb_height〉, 〈class〉`

- 〈video_id〉 is the video numeric identifier, starting with 1. It represents the position of the video in the list of all videos, sorted in alphanumeric order.
- 〈frame〉 represents the frame count for the current frame in the current video, starting with 1.
- 〈tl_x〉 is the x-coordinate of the top left point of the bounding box.
- 〈tl_y〉 is the y-coordinate of the top left point of the bounding box.
- 〈bb_width〉 is the width of the bounding box.
- 〈bb_height〉 is the height of the bounding box.
- 〈class〉 is the class id of the object as given in the labels information above.


## Usage

To use this project for training and post-processing with YOLOv8, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/SiavashShams/Helmet-Rule-Violation-Detection-YOLO.git
   cd Helmet-Rule-Violation-Detection-YOLO
   ```

2. To train the model, run the following from the `src` directory:
   ```
   cd src
   python train_yolov8x.py
   ```

3. For post-processing, execute the provided script:
   ```
   python postprocessing.py
   ```


## Folder Structure

- `/config`: Configuration files and scripts for setting up the project environment.
- `/src`: Source code of the project required for training YOLO models and postprocessing.
- `Assignment2_report.pdf`: Detailed report of the project findings and methodology.
- `assignment2.ipynb`: Jupyter notebook with all the code, comments, and explanations regarding the project's analysis and results.

### Dataset/Model Google Drive Link:
Data and models:https://drive.google.com/drive/folders/1Ri24mR17sl9ifj-8gzgQc1r-J-uIGK14?usp=sharing


