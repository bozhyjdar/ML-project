# Используем официальный легкий образ Python
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p plots

CMD ["python", "main.py"]