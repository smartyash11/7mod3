# Use an official lightweight Python image as a base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependencies file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (main.py and fetchers.py) into the container
COPY ./app /app

# The command to run the ingestion script when the container starts.
# This makes the container a one-off task runner.
CMD ["python", "main.py"]