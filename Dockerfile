FROM jupyter/tensorflow-notebook:45bfe5a474fa

RUN conda install -y pygraphviz
RUN pip install graphviz

