# Sandbox image for agent-generated code.
# Build (from the project root):
#   docker build -t eda-sandbox:1.0 -f sandbox/Dockerfile sandbox
#
# Only the Python environment lives in the image. worker.py and data.py are
# bind-mounted read-only at run time (see docker_sandbox.py), so code changes
# never require a rebuild and can never drift from the host copy.
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    MPLCONFIGDIR=/tmp/mpl \
    MPLBACKEND=Agg \
    HOME=/tmp \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-sandbox.txt /tmp/requirements-sandbox.txt
RUN pip install --no-cache-dir -r /tmp/requirements-sandbox.txt \
 && rm /tmp/requirements-sandbox.txt

# Non-root user with a fixed uid; /output is the only writable data path
# (a bind mount at run time). /data is mounted read-only at run time.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox \
 && mkdir -p /opt/eda /data /output \
 && chown sandbox:sandbox /output

USER 10001:10001
WORKDIR /output
CMD ["python", "-u", "/opt/eda/worker.py"]
