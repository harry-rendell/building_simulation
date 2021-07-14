FROM jupyter/scipy-notebook
LABEL maintainer="Paolo Di Prodi"
LABEL description="Nomad PySpice"

USER root

RUN apt-get update && apt-get install -y \
  libngspice0-dev \
  ngspice \
  && rm -rf /var/lib/apt/lists/*
  

#RUN chown ${NB_USER} /usr/local/share/jupyter/kernels/pyspark3/kernel.json \
#    && chown -R ${NB_USER}:${NB_USER} ${HOME} ${JUPYTER_DIR}

WORKDIR ${HOME}

USER ${NB_USER}

RUN python3 -m pip install PySpice --user

RUN git clone https://github.com/FabriceSalvaire/PySpice.git