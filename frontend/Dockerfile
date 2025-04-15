# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn (production WSGI server)
RUN pip install --no-cache-dir gunicorn

# Copy the rest of the application code (frontend app, templates, static)
COPY . .

# Expose the port the app runs on (should match the CMD)
EXPOSE 8080

# Define the command to run the application using Gunicorn
# Use the PORT environment variable provided by Cloud Run
# Frontend usually needs fewer resources, adjust workers/threads if needed
# Use 0.0.0.0 to accept connections from outside the container
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 0 app:app
