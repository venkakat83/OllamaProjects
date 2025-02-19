FROM python:3.11-slim

# Set the working directory
WORKDIR /flask-app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "src/backend.py"]