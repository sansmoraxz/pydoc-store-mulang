ARG WORKDIR="/app"

FROM python:3.11-buster as builder
ARG WORKDIR
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
RUN pip install poetry && poetry config virtualenvs.in-project true
WORKDIR ${WORKDIR}
COPY . .
RUN poetry install --no-interaction --no-ansi -vvv

FROM nvidia/cuda:11.6.2-base-ubuntu20.04 as release
ARG WORKDIR
WORKDIR ${WORKDIR}
COPY --from=builder ${WORKDIR} .
ENTRYPOINT [ "./.venv/bin/python", " --version"]
