FROM ubuntu:xenial AS ubuntu16_04-with-python3_6
LABEL author="gaoxiang"
# 用ubuntu国内源替换默认源
RUN rm /etc/apt/sources.list
COPY sources.list /etc/apt/sources.list
# 安装python3.6必要的包。源镜像太精简了，ip ifconfig之类的都没有。后续安装python pip也需要一些。但是build_essential似乎不必须，先去了。如果后面安装numpy之类需要gcc了，再加上
RUN apt-get update \
    && apt-get install -y libapt-pkg5.0 apt-transport-https iproute2 net-tools ca-certificates curl wget software-properties-common \
    && apt-get clean
# 安装python3.6 来自第三方
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.6 python3.6-dev python3-pip \
    && apt-get clean
# 和自带的3.5共存
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2 \
    && update-alternatives --config python3
# 更改 pip 源为清华源，加速
RUN mkdir /root/.pip
COPY pip.conf /root/.pip/pip.conf
# 更新 pip3
RUN pip3 install --upgrade pip \
    && rm -rf /root/.cache/pip
#print()时在控制台正常显示中文
ENV PYTHONIOENCODING=utf-8

