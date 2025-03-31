FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure Python can find your modules
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command - runs a basic example
CMD ["python", "-m", "src/main.py"]