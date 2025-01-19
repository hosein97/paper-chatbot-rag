# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (for example, for Chroma, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your backend will run on
EXPOSE 8000

# Copy the .env file and use it in the container
COPY .env .env

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
