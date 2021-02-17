FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN mkdir /app 
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN python3 -m pip install gevent
RUN python3 -m pip install pillow
ENTRYPOINT ["python3"]
CMD ["app.py"]
