FROM python:3.9-slim

# Build argument for port with default value
ARG PORT=8001
ENV PORT=${PORT}
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt . 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port from build argument
EXPOSE ${PORT}

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Default command (can be overridden by Docker Compose)
CMD ["sh", "-c", "${CMD_TO_RUN}"]
