# Use the official Python image with the specified version
FROM python:3.12.4

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1  # Prevents Python from writing .pyc files
ENV PYTHONUNBUFFERED 1         # Ensures stdout/stderr is unbuffered

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install dependencies with trusted hosts to avoid SSL issues
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --trusted-host download.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org


# Copy the Django application code to the working directory
COPY . /app/

# Expose the port Django will run on
EXPOSE 8000

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
