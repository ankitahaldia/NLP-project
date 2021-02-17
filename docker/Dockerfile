FROM ubuntu:latest
FROM python

RUN mkdir /app
RUN mkdir /app/docker
RUN mkdir /app/pipeline

COPY docker /app/docker
COPY pipeline /app/pipeline
COPY pipeline/model /app/pipeline/model
COPY pipeline/preprocess /app/pipeline/preprocessing
COPY app.py /app/app.py

WORKDIR /app

CMD ["python","app.py"]