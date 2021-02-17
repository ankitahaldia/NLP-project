FROM python


RUN mkdir /app
RUN mkdir /app/docker

COPY docker /app/docker

COPY app.py /app/app.py

WORKDIR /app

CMD ["python","app.py"]