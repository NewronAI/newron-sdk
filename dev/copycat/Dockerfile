FROM alpine:3.10

RUN apk add --no-cache bash
RUN apk add --no-cache git
ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "-c", "/entrypoint.sh"]
