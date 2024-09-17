# Use Python 3.10.14 Alpine as the base image
FROM python:3.10.14-alpine3.20

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src/api /app

# Install any needed packages specified in requirements.txt
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]