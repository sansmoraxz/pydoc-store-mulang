ARG WORKDIR="/app"

FROM nvidia/cuda:11.6.2-base-ubuntu20.04
ARG WORKDIR
WORKDIR ${WORKDIR}
COPY . .
ENTRYPOINT [ "bash"]
