FROM python:3.12-alpine

RUN apk add --no-cache ca-certificates curl docker-cli tzdata

WORKDIR /app
COPY openclaw_monitor.py /app/openclaw_monitor.py
COPY probe_contract.py /app/probe_contract.py
COPY reporting_time.py report_outbox.py /app/
COPY reporting_projection.py /app/
COPY host_metrics_contract.py /app/

EXPOSE 18080

CMD ["python", "/app/openclaw_monitor.py"]
