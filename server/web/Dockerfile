FROM python:3.8-slim-buster

WORKDIR /app

EXPOSE 5000

RUN apt-get update -qq
RUN apt-get install -y --no-install-recommends gcc build-essential git libsndfile1-dev -qq

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]