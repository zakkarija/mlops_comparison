FROM python:3.7.2

RUN rm /etc/apt/sources.list
RUN echo "deb http://archive.debian.org/debian-security jessie/updates main" >> /etc/apt/sources.list.d/jessie.list
RUN echo "deb http://archive.debian.org/debian jessie main" >> /etc/apt/sources.list.d/jessie.list
RUN apt-get update
RUN pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
COPY src /app/
WORKDIR /app/

CMD ["python", "./main.py" ]