FROM python:3.10.14

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy all source code (includes main.py, classes/, helpers/, config/)
COPY src/ /app/src/

# Create output directory
RUN mkdir -p /app/output

# Set working directory to src
WORKDIR /app/src

# Default command
CMD ["python", "main.py"]