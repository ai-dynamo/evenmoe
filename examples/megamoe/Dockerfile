# SPDX-License-Identifier: Apache-2.0
FROM lmsysorg/sglang@sha256:069efa1d6a2d58309529f53eeee34c39d69b1cdb455eaf7fb8de83e3fd0be727
WORKDIR /opt/evenmoe
COPY . /opt/evenmoe
ENV FLASHINFER_CUDA_ARCH_LIST=9.0a
RUN python3 -m pip install 'setuptools-scm>=8' 'setuptools-rust>=1.11' && \
    python3 examples/megamoe/build_runtime.py && \
    python3 -m pip install --no-deps . && \
    python3 -m compileall -q examples/megamoe/runtime
ENTRYPOINT ["python3", "/opt/evenmoe/examples/megamoe/launch.py"]
