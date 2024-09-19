# Use Python 3.10.14-slim as the base image
FROM python:3.10.14-slim

# Set the working directory in the container
WORKDIR /app

# Update the package list and install necessary dependencies
RUN apt-get update; \
    apt-get install -y \
        build-essential pandoc pkg-config libhdf5-serial-dev; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*;

# Copy the current directory contents into the container at /app
COPY src/api /app/
COPY src/scripts /app/scripts/
COPY models /app/models/
COPY data/preprocessed /app/data/preprocessed/

# Upgrade pip to ensure it is up-to-date
RUN python -m pip install --upgrade pip

# Install poetry
# The --no-cache-dir flag tells pip to not use its cache for downloading and installing packages, ensuring a clean installation.
RUN pip install --no-cache-dir poetry

# Copy only pyproject.toml and poetry.lock (if it exists)
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
# `poetry config virtualenvs.create false`: This configures Poetry to not create a virtual environment.
# In Docker, it's generally best practice to install dependencies directly in the container's environment rather than using virtual environments.
# `poetry install --no-interaction --no-ansi`: This tells Poetry to install all project dependencies listed in the pyproject.toml file.
# The flags disable interactive prompts and ANSI color codes for clean output in the build process.
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Download punkt_tab used in nltk
RUN python -m nltk.downloader punkt_tab

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Command to run the FastAPI server
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]