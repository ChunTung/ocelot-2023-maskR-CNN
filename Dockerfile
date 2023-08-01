#從網路上尋找映像檔
FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PATHONPATH "${PYTHONPATH}:/opt/app/"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
#Run pip install --upgrade pip

RUN pip install pillow
RUN pip install opencv-python-headless==4.5.5.62
RUN pip install scikit-image==0.17.2 
RUN pip install scipy==1.5.4
RUN pip install matplotlib==3.3.4
RUN pip install multiprocess
RUN pip install tqdm==4.64.1
RUN pip install keras==2.2.4
RUN pip install pandas==1.1.5

COPY ./ /opt/app/

COPY --chown=user:user process.py /opt/app/



ENTRYPOINT ["python", "-m","process"]

