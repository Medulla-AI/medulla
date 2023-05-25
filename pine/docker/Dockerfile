FROM debian AS base
USER root
RUN apt-get update && apt-get upgrade && apt-get install python3 python3-pip git -y
RUN pip3 install --upgrade pip

RUN git clone https://github.com/HMUNACHI/PINE

WORKDIR /PINE

COPY ../requirements .

RUN pip3 install -r requirements_cpu.txt

CMD ["bash"]
