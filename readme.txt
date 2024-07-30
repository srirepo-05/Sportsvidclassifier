                                                SPORTS DATA CLASSIFICATION


This project predicts the sports in a given sports video. It uses Yolov5 model for classification. Currently this project
predicts five sports namely CRICKET, BASEKETBALL, SWIMMING, BUNJEE JUMPING, TABLE TENNIS. We are focussing on our project to
expand its prediction for more than five sports and even we are planning to put it in use for a real time application. 
If we input a sports video and pass it through the program, it generates a output video of the inputted sports video
which we uploaded along with PREDICTION, POSE ESTIMATION, FEATURES(angle).

Table of content :-

1)Required dependencies and datasets
2)Installation of dependencies and import of datasets
3)Installation of python virtual environment
4)How to use the project
5)References
6)Credits
7)License

REQUIRED DEPENDENCIES AND DATASETS :-


* Pytorch
* Mediapipe
* Opencv
* Pandas
* Matplotlib
* Numpy
* Sports Dataset 
* Yolov5


Installation of dependencies and import of datasets :-



Pytorch -   pip install pytorch torchvision torchaudio ( if you want to make use of GPU refer installation with CUDA in 
            pytorch website)

Mediapipe - pip install mediapipe

Opencv -    This module will be installed automatically when you install mediapipe ( incase if you find any module missing
	    regarding opencv run the command - pip install opencv-python on terminal).

Pandas -    pip install pandas

Matplotlib - pip install matplotlib

Sports dataset - git clone https://github.com/Susa-43/Sports_Classification_dataset.git

Yolov5 - git clone https://github.com/ultralytics/yolov5.git



Installation of Python virtual environment :- 


If the project is done using the real Python environment, we would be installing more number of modules that makes our 
real environment more clumsy. So it is preferable to use a python virtual environment for the project. Here are the commands
to create Python virtual environment.

Windows :-

> py -m venv project (to create virtual enviroment)
> .\project\Scripts\activate (to activate the virtual environment)

Mac/linux :-

$ py -m venv project (to create virtual environment)
$ source tfod/bin/activate (to activate the virtual environment)



How to use the project :-


In the command given in prediction part of the code, give the path for the sports video you want to predict. The output video
will be created automatically and saved to the current working directory.



References:- 

Yolov5 classification - https://github.com/ultralytics/yolov5/discussions/8996
Dataset - www.kaggle.com
Creation of dataset - app.robowflow.com
To train Yolov5 for custom dataset - https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
Yolov5 classification - https://youtu.be/PwIQc06gnCI
Feature extraction - https://youtu.be/06TE_U21FK4
Video Writer - https://www.geeksforgeeks.org/saving-a-video-using-opencv/
Graph plotting - https://youtu.be/0P7QnIQDBJY

Credits :-

Sriram M K - Frame conversion, Graph generation 
Sabari Srinivas - Feature extraction, skeletonization
Rathish Manivannan - Graph generation, Image to video conversion
Sudharsanan K - Classification
Rishith Reddy - Skeletonization
Guru Yogendra Reddy - Graph generation, Frame conversion
Sabesh Raaj - Feature extraction, Frame conversion
Harsha Vardhana Anand - Dataset gathering, classification


License :-

This is an open source project.


*Note - Use google colab or jupyter to open the project file
*Note - If you want to train different datasets or more labels, change the path for dataset in the training part of the code.









