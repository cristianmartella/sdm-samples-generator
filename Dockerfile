FROM python:3.13-alpine AS builder

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


FROM builder

COPY src/main ./src

CMD [ "python", "./src/app.py" ]