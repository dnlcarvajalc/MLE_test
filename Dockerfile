FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements_docker.txt .
RUN pip install --upgrade pip && pip install -r requirements_docker.txt

COPY src ./src
COPY output ./output

EXPOSE 8000

CMD ["uvicorn", "src.e_apification:app", "--host", "0.0.0.0", "--port", "8000"]
