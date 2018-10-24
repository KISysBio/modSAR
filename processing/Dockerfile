FROM jupyter/all-spark-notebook:50247fea877a

ENV PATH="/opt/cplex:${PATH}"
WORKDIR /mnt/code

RUN conda install -y -c rdkit rdkit
RUN conda install -y -c conda-forge python-igraph

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN ipython3 profile create
RUN echo 'import pandas as pd; pd.set_option("display.width", 160)' >> /home/jovyan/.ipython/profile_default/ipython_config.py

USER root
RUN apt update
RUN apt install -y openjdk-8-jdk

USER jovyan

RUN cd ~ && git clone https://github.com/jonjoncardoso/cdk_pywrapper.git
RUN cd ~/cdk_pywrapper && pip install .

RUN jupyter contrib nbextension install --user
RUN jupyter nbextensions_configurator enable --user
