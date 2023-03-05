# Object_Detection_Yolo
The model is trained for Car, Truck and Van detection.

Prerequisites
Tensorflow
opencv-python

There are many ways to install virtual environment (virtualenv), see the Python Virtual Environments: A Primer guide for different platforms, but here are a couple:

For Ubuntu

     pip install virtualenv
For Mac

     pip install --upgrade virtualenv

Create a Python 3.7 virtual environment for this project and activate the virtualenv:

    virtualenv -p python3.6 yolo_object

    source ./yoloface/bin/activate

Usage
Clone this repository

     git clone https://github.com/rajban94/Object_Detection_Yolo/
For Object detection, you should change the model_path with your path in config.ini

Run the following command:

image input

    python controller.py --file 'image path'
