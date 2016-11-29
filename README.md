# PatRec - Pattern Recognition for Logo Matching

PatRec is a Flask Web App that allows you to scan a set of images for a given pattern. 

![screenshot](https://github.com/batzner/patrec/raw/master/img/screenshot.png)

It uses an ensemble of Feature Descriptors to identify the pattern in each of the given images. Thus it achieves a high accuracy without having to train the ensemble.

## Installation
There are three options to run PatRec in your browser:

#### 1. Docker (Docker Hub) - (recommend)
Run `docker run -p 5000:80 jorba/patrec`. This will automatically pull the latest patrec image from Docker Hub (http://hub.docker.com).
The image size is 3 GB. This option is the fastest way if you do not have OpenCV 2.4.X installed.

#### 2. Docker (local)
1. Build the image by running `docker build -t patrec .` (Don't forget the point at the end of the command) in the project's root directory where the `Dockerfile` is located. However, building the Dockerimage will take a very long time.
2. Run the image with `docker run -p 5000:80 patrec`

#### 3. Install without Docker (Linux)
1. Install all python dependencies by running `sudo pip install -r requirements.txt` in the project's root directory.
2. Install OpenCV 2.4.13 on your machine following [these steps](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html).
3. Start the Flask web app with `python runlocal.py`.

#### 4. Install without Docker (Windows)
1. (Recommended) Install the [Anaconda Python Distribution](https://www.continuum.io/downloads)
2. Install all python dependencies by running `pip install -r requirements.txt` in the project's root directory.
3. Install OpenCV 2.4.13 on your machine by downloading the OpenCV Installer from the [OpenCV Website](http://opencv.org/). 
4. Navigate to the folder `<Your_OpenCV_Path>/build/python/2.7` and copy the file `cv2.pyd` to your python 2.7 site-packages folder.
5. To check if the installation was successful run `python`, `import cv2` and `cv2.__version__`. The version should be `2.4.13`.
6. Finally, start the Flask web app with `python runlocal.py` from the application folder.

## Usage
To try out PatRec, go to `localhost:5000` and drop the images and the pattern in the boxes. Then click `Find matches` to see the result. 

The result will contain the number of votes from the Feature Descriptors and the corresponding prediction: `Match` or `No Match`.
