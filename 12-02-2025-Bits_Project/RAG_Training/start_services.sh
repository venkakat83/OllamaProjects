#!/bin/sh

# Start Ollama service
ollama start &

# Run the Python script
python SchneiderProductContentTraining.py