FROM python:3.10.14
 
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
COPY src /app/
WORKDIR /app/
 
CMD ["python", "./main.py" ]