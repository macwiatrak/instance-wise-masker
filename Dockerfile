FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# Install Python requirements
RUN pip3 install --upgrade --compile --no-cache pip wheel setuptools

RUN mkdir /instance-wise-masker
WORKDIR /instance-wise-masker

COPY requirements.txt /instance-wise-masker/requirements.txt
COPY . /instance-wise-masker

RUN pip3 install -r requirements.txt
RUN pip3 install -e .