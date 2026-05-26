FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Install system dependencies (if any needed for pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Expose port (Render uses $PORT env var, default 8000)
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "backend/app/main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
