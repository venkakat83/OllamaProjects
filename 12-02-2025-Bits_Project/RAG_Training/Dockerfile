# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama (example, adjust as needed)
RUN apt-get update && apt-get install -y wget

# Download and install Ollama (example, adjust URL and installation steps as needed)
RUN wget -O ollama.deb https://ollama.com/download/ollama.deb && \
    dpkg -i ollama.deb && \
    apt-get install -f && \
    rm ollama.deb

# Make port 80 available to the world outside this container
EXPOSE 80

# Copy the start_services.sh script into the container
COPY start_services.sh /app/start_services.sh

# Make the script executable
RUN chmod +x /app/start_services.sh

# Define environment variable

#ENV NAME World

#The above line "ENV NAME World" in your Dockerfile sets an environment variable named NAME with the value World inside the Docker container. 
#This environment variable can be accessed by the application running inside the container. 
#For example, in your Python script, you could retrieve this value using os.environ['NAME'].

# Use the shell script as the CMD
CMD ["/app/start_services.sh"]