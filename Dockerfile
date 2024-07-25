FROM python:3.10.11

WORKDIR /usr/src/app
COPY . .

RUN python -m venv venv
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt
RUN . venv/bin/activate && python train.py

CMD ["sh", "-c", ". venv/bin/activate && python main.py"]   