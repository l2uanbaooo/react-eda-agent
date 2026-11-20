# Sandbox image for agent-generated code.
# Build from the project ROOT (the context is the root; .dockerignore lets only
# the two requirement files in):
#   docker build -t eda-sandbox:1.1 -f sandbox/Dockerfile .
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

# Direct pins + the host lock file as constraints: every package that exists on
# both sides (including indirect ones such as tzdata) gets the host version.
COPY sandbox/requirements-sandbox.txt requirements-lock.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-sandbox.txt -c /tmp/requirements-lock.txt \
 && rm /tmp/requirements-sandbox.txt /tmp/requirements-lock.txt

# Non-root user with a fixed uid; /output is the only writable data path
# (a bind mount at run time). /data is mounted read-only at run time.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox \
 && mkdir -p /opt/eda /data /output \
 && chown sandbox:sandbox /output

USER 10001:10001
WORKDIR /output
CMD ["python", "-u", "/opt/eda/worker.py"]
