FROM python:3.12.3-slim-bullseye

USER root

RUN mkdir -p /home/courses/cmpt353/project/
WORKDIR /home/courses/cmpt353/project/

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY data ./
COPY .env ./

# CMD ["..."]
# RUN ...