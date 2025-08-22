# Use an official lightweight Python image as a base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependencies file first to leverage Docker's layer caching.
# This step is only re-run if requirements.txt changes.
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container.
# This includes the main.py, ml_utils.py, and the model/ directory.
COPY ./app /app

# The command to run the Uvicorn server for the FastAPI application when the container starts.
# It will be accessible on port 8002 inside the container's network.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]