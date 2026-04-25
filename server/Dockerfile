FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
