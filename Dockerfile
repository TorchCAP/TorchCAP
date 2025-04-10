FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt update
RUN apt install -y libeigen3-dev swig
RUN apt install -y graphviz

RUN pip install --upgrade pip setuptools
RUN pip install tabulate ortools
RUN pip install transformers==4.48.2
RUN pip install pwlf
