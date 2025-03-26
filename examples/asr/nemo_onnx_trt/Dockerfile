FROM nvcr.io/nvidia/tritonserver:24.08-py3

RUN pip install git+https://github.com/NVIDIA/NeMo.git@v2.2.0rc3#egg=nemo_toolkit[asr]
RUN pip install tensorrt==10.0.1

COPY . /workspace

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

CMD ["tritonserver", "--model-repository=/workspace/model_repository"]