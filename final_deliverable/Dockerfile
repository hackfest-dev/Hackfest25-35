FROM python:3.12-slim

WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY packages.txt .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y $(cat packages.txt) && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and models
COPY . .

# Set environment variables
ENV PORT=8501

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
