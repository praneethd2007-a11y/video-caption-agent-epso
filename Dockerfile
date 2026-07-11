
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

#installing python dependencies
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ai_inference.py video_utils.py ./

ENV TASKS_PATH=/input/tasks.json
ENV RESULTS_PATH=/output/results.json

# What runs when the container starts
CMD ["python", "main.py"]