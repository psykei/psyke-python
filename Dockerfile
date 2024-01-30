FROM python:3.9
ARG PSYKE_VERSION
EXPOSE 8888
RUN apt update; apt install -y -q openjdk-17-jdk
RUN pip install jupyter
RUN pip install psyke==$PSYKE_VERSION
RUN mkdir -p /root/.jupyter
ENV JUPYTER_CONF_FILE /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_origin = '*'" > $JUPYTER_CONF_FILE
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> $JUPYTER_CONF_FILE
RUN mkdir -p /notebook
COPY test/resources/datasets/*.csv /notebook/datasets/
WORKDIR /notebook
CMD jupyter notebook --allow-root --no-browser
