FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--host", "0.0.0.0", "--port", "8000"]
