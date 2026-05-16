ARG TORCH_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

ARG TORCH_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION

# Fix nvidia-key error issue (NO_PUBKEY A4B469963BF863CC)
RUN rm /etc/apt/sources.list.d/*.list

# Installing apt packages
RUN export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	  git wget tmux vim zsh zip build-essential python3-dev cmake ninja-build libopenblas-dev libsparsehash-dev \
	  gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 \
	  libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 \
	  libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 \
	  libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils \
      ffmpeg libsm6 libxext6   -y \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

# Install Pointcept environment
# RUN conda run -n pointcept
RUN conda install h5py pyyaml -c anaconda -y
# RUN conda install tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y 
#RUN conda install libffi==3.3 -y

RUN pip3 install --upgrade pip && \
    pip3 install torch-geometric spconv-cu$(echo ${CUDA_VERSION} | tr -d ".0") open3d

ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}

RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git
WORKDIR /workspace/MinkowskiEngine
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9" python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas \
	&& cd /workspace \
 	&& rm -r MinkowskiEngine
	
WORKDIR /workspace
#RUN git clone https://github.com/Karbo123/segmentator.git \
#    && cd segmentator/csrc \
#    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
#    && mkdir build \
#    && cd build \
#    && cmake .. \
#        -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
#        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
#        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
#        -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` \
#    && make \
#    && make install \
#    && cd ../../..

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6"

# Build pointops
RUN git clone https://github.com/Pointcept/Pointcept.git
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6"  pip install Pointcept/libs/pointops -v --no-cache-dir --no-build-isolation

# Build pointgroup_ops
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6" pip install Pointcept/libs/pointgroup_ops -v --no-cache-dir --no-build-isolation 

RUN TORCH_CUDA_ARCH_LIST="8.0 8.6"  pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-cache-dir --no-build-isolation 

# Build swin3d
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6" pip install -U git+https://github.com/microsoft/Swin3D.git -v --no-cache-dir --no-build-isolation 

WORKDIR /workspace
RUN git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && git checkout 22c0358f4ba7999a15dbe27989ce163e5edb5693
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6" pip3 install ./flash-attention --no-build-isolation --no-cache-dir --no-build-isolation 

# WORKDIR /tmp
COPY requirements.txt .
RUN pip3 install -r requirements.txt

    

