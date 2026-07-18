# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

# Install uv (dependency manager)
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Command to run API with Uvicorn
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
