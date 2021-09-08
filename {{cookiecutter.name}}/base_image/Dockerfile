FROM python:3.8

COPY ./dist/* /dist/

ADD requirements.txt /

RUN apt-get update && \
  pip install -r /requirements.txt && \
  pip install  /dist/*

ENTRYPOINT [ "python" ]
CMD [ "-m", "trainer.main" ]