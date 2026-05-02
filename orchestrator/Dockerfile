# Orchestrator image: FastAPI + numpy/scipy/sklearn/networkx for the GSPDA
# policy library. No TensorFlow — the orchestrator never runs the model.

FROM python:3.10-slim

WORKDIR /app

COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The orchestrator imports the policy library, so we need to bring it
# into the image. The user package isn't needed here.
COPY policy /app/policy
COPY orchestrator /app/orchestrator

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "-m", "orchestrator.main"]
