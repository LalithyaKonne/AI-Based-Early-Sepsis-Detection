FROM python:3.12-slim

# Set working directory to /app/backend
WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python deps
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Environment variables (Render will override)
ENV MODEL_PATH=/app/backend/models/final_sepsis_model.pkl
ENV JWT_SECRET=super-secret-key

# Expose port (Render provides $PORT)
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
