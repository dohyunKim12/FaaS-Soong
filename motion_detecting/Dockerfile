FROM ubuntu:18.04

RUN apt-get update -y && apt-get install -y python3-pip

RUN apt-get install -y libgl1-mesa-glx

WORKDIR /app

RUN python3 -m pip install pip --upgrade

RUN pip3 install cmake && pip3 install scikit-build && pip3 install opencv-python && pip3 install numpy

RUN python3 -m pip install pika --upgrade

COPY . /app

CMD ["python3", "motion_detect.py"]
