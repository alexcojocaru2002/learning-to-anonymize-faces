# Learning-to-anonymize-faces
This repository contains the code for a reproduction of the paper "Learning to Anonymize Faces for Privacy Preserving Action Detection" presented at ECCV 2018.

## Installation
1. Clone the repository

```sh
git clone https://github.com/alexcojocaru2002/learning-to-anonymize-faces.git
```

2. Install the required libraries (make sure you have Python and pip already installed)
**Optional**: If you have CUDA, first check its version by running 
```sh
nvcc --version
```
and then go to https://pytorch.org/get-started/locally/ and run the command suggested there to ensure the torch versions maches the CUDA version.

Then proceed with installing the rest of the dependencies:


```sh
pip install -r requirements.txt
```

3. Download the required data

First you should create the outputs/, data/ folders. 

To use **JHMDB** you have to download the .tar.gz archive from https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct.

Afterwards, place the JHMDB folder in the /data folder.


4. Download the weights for the pretrained models

For the pretrained Yolov8-face model we use the weights from here https://github.com/derronqi/YOLOv8-Face?tab=readme-ov-file (the yolov8n model)
For the pretrained sphereface model we use the weights from here: https://github.com/clcarwin/sphereface_pytorch/blob/master/model/sphere20a_20171020.7z

## Data Visualization 

To run an example script you can use the following command that runs the visualize_data script to save the frames for one video from JHMDB. 

```sh
python main.py visualize_data
```
