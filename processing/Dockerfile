FROM jupyter/scipy-notebook:50247fea877a

ARG CDK_VERSION=2.2
ARG JAR_PATH=/opt

ENV PATH="/opt/cplex:${PATH}"
WORKDIR /mnt/code

RUN conda install -y -c rdkit rdkit
RUN conda install -y -c conda-forge python-igraph
RUN conda install -y py4j=0.10.8.1

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN ipython3 profile create
RUN echo 'import pandas as pd; pd.set_option("display.width", 160)' >> /home/jovyan/.ipython/profile_default/ipython_config.py

USER root
RUN apt update
RUN apt install -y openjdk-8-jdk
RUN wget "https://github.com/cdk/cdk/releases/download/cdk-${CDK_VERSION}/cdk-${CDK_VERSION}.jar" \
  --directory-prefix=${JAR_PATH}

USER jovyan

RUN jupyter contrib nbextension install --user
RUN jupyter nbextensions_configurator enable --user
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter nbextension enable execute_time/ExecuteTime
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable codefolding/edit
RUN jupyter nbextension enable toc2/main
RUN jupyter nbextension enable collapsible_headings/main
