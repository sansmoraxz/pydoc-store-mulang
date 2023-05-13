ARG WORKDIR="/app"

FROM python:3.11-buster as builder
ARG WORKDIR
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR ${WORKDIR}
RUN pip install poetry
RUN poetry config --local virtualenvs.in-project true \
    && poetry config --local virtualenvs.options.always-copy true
COPY . .
RUN poetry install --no-interaction --no-ansi -vvv

FROM nvidia/cuda:11.6.2-base-ubuntu20.04 as release
ARG WORKDIR
WORKDIR ${WORKDIR}
COPY --from=builder ${WORKDIR} .
ENTRYPOINT [ "bash"]
