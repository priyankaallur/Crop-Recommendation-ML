FROM python:3.11-slim

WORKDIR /app

# Install system deps for building some packages (if needed)
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

ENV PORT=5000
EXPOSE 5000

# Run using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
