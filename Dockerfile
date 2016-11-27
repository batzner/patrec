FROM ubuntu

MAINTAINER Felix Schober <mail@felix-schober.de>

# update repositories
RUN apt-get update
RUN apt-get -y upgrade

# install developer tools
RUN apt-get -y install build-essential cmake pkg-config git unzip wget python2.7-dev python-scipy

# install image/video and other important packages / tools for opencv
RUN apt-get -y install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get -y install  libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

# get pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN python -m pip install --upgrade pip

# compile opencv 2.4.X from source
RUN wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.13/opencv-2.4.13.zip
RUN unzip opencv-2.4.13.zip
RUN mkdir opencv-2.4.13/build
WORKDIR /opencv-2.4.13/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
RUN make
RUN make install

# clean up 
WORKDIR /
RUN rm -r opencv-2.4.13
RUN rm opencv-2.4.13.zip

# load patrec
RUN mkdir patrec
WORKDIR /patrec/
RUN git init
RUN git pull https://github.com/batzner/patrec.git

# install other python packages
RUN pip install -r requirements.txt

# Start Server when running the container
ENTRYPOINT ["python", "runserver.py"]

# Expose port 80 so that the server can be accesed by http://127.0.0.1
EXPOSE 80
