# PatRec - Pattern Recognition for Logo Matching

PatRec is a Flask Web App that allows you to scan a set of images for a given pattern. 

![screenshot](https://github.com/batzner/patrec/raw/master/img/screenshot.png)

It uses an ensemble of Feature Descriptors to identify the pattern in each of the given images. Thus it achieves a high accuracy without having to train the ensemble.

## Installation
There are three options to run PatRec in your browser:

#### 1. Docker (Docker Hub)
Run `docker run -p 5000:80 jorba/patrec`. The image is 3 GB large.

#### 2. Docker (local)
Build the image by running `docker build` in the project's root directory.

#### 3. Install without Docker
1. Install all python dependencies by running `sudo pip install -r requirements.txt` in the project's root directory.
2. Install OpenCV 2.4.13 on your machine following [these steps](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html).
3. Start the Flask web app with `python runlocal.py`.

## Usage
To try out PatRec, go to `localhost:5000` and drop the images and the pattern in the boxes. Then click `Find matches` to see the result. 

The result will contain the number of votes from the Feature Descriptors and the corresponding prediction: `Match` or `No Match`.
