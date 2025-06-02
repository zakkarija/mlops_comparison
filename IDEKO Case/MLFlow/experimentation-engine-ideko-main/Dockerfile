
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Update
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    unzip \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Java 13
RUN apt-get install -y openjdk-13-jdk

# Install Python 3.9 and pip
RUN apt-get update --fix-missing && apt-get upgrade -y
RUN apt-get install -y python3.9 python3.9-distutils python3-pip python3.9-venv

# Install Java 13
RUN apt-get install -y openjdk-13-jdk

# Set Java enronment variables
ENV JAVA_HOME=/usr/lib/jvm/java-13-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Creat workdir and copy the code inside
WORKDIR /app
COPY . /app

# Create the environment
RUN python3.9 -m venv env

# Update pip and setuptools inside the env
RUN ./env/bin/pip install --upgrade pip setuptools wheel python-dotenv

# Install exp_engine inside the env
RUN ./env/bin/pip install eexp_engine

# Run it on the concrete environment
CMD ["./env/bin/python", "example.py"]