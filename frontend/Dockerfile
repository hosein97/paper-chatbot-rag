# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Gradio will run on
EXPOSE 7860

# Define environment variable for Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0

# Run Gradio app
CMD ["python", "app.py"]
