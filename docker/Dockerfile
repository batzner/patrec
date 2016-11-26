FROM ubuntu

MAINTAINER Felix Schober <mail@felix-schober.de>

# update repositories
RUN apt-get update
RUN apt-get --yes --force-yes upgrade

# install developer tools
RUN apt-get --yes --force-yes install build-essential cmake pkg-config
RUN apt-get --yes --force-yes install git

# install image/video and other important packages / tools for opencv
RUN apt-get --yes --force-yes install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
RUN apt-get --yes --force-yes install libgtk2.0-dev
RUN apt-get --yes --force-yes install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get --yes --force-yes install libatlas-base-dev gfortran
RUN apt-get --yes --force-yes install unzip

# get pip
RUN apt-get --yes --force-yes install wget
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN python -m pip install --upgrade pip

RUN apt-get --yes --force-yes install python2.7-dev


# install other python packages
RUN apt-get --yes --force-yes install python-scipy
RUN apt-get --yes --force-yes install python-opencv
RUN apt-get install -y nginx supervisor

RUN pip install click
RUN pip install Flask
RUN pip install itsdangerous
RUN pip install Jinja2
RUN pip install MarkupSafe
RUN pip install Pillow
RUN pip install scikit-learn
RUN pip install sklearn
RUN pip install Werkzeug
RUN pip install Pillow

# load patrec
WORKDIR /
RUN mkdir patrec
WORKDIR /patrec/
RUN git init
RUN git pull https://github.com/batzner/patrec.git



# run server
#CMD ...





# Set default container command
ENTRYPOINT ["python", "runserver.py"]

# Expose the default Flask Port
EXPOSE 5000
