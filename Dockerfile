# Use Python image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
