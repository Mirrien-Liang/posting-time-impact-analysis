FROM python:3.12.3-slim-bullseye

RUN mkdir -p /home/courses/cmpt353/project/
WORKDIR /home/courses/cmpt353/project/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY .env.docker ./.env
COPY data src ./

CMD ["python", "-m", "src.pipeline"]
