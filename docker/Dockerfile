FROM ubuntu:latest
FROM python

COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt
RUN mkdir /app
RUN mkdir /app/docker
RUN mkdir /app/pipeline

COPY docker /app/docker
COPY pipeline /app/pipeline
COPY pipeline/model /app/pipeline/model
COPY pipeline/preprocess /app/pipeline/preprocessing
COPY stream_app.py /app/stream_app.py

WORKDIR /app

RUN streamlit run stream_app.py

CMD ["python","stream_app.py"]