FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "backend" ]; then\n\
    echo "Starting simple backend..."\n\
    cd /app && python backend/simple_main.py\n\
elif [ "$1" = "frontend" ]; then\n\
    echo "Starting frontend..."\n\
    cd /app && streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0\n\
else\n\
    echo "Usage: docker run <image> [backend|frontend]"\n\
    exit 1\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 8002 8501

# Default command
CMD ["/app/start.sh", "backend"]