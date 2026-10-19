FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY code /app/code
RUN python -m pip install --no-cache-dir -r /app/code/requirements.txt
COPY data/examples /app/data/examples
COPY data/templates /app/data/templates
COPY data/official /app/data/official
COPY data/benchmarks /app/data/benchmarks
COPY data/benchmark_index.json /app/data/benchmark_index.json
RUN mkdir -p /app/state && chown -R 10001:10001 /app/state
ENV TRACECODER_STATE_DIR=/app/state
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=3s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=2)"]
CMD ["python", "code/app.py", "--host", "0.0.0.0", "--public-demo"]
