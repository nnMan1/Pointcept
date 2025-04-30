ARG TORCH_VERSION=2.0.1
ARG CUDA_VERSION=11.7
ARG CUDNN_VERSION=8

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
	  git wget tmux vim zsh build-essential cmake ninja-build libopenblas-dev libsparsehash-dev \
	  gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 \
	  libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 \
	  libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 \
	  libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils \
      ffmpeg libsm6 libxext6  -y \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

# Install Pointcept environment
# RUN conda install h5py pyyaml -c anaconda -y
# RUN conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
# RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y 
# RUN conda install libffi==3.3 -y

# RUN pip3 install --upgrade pip && \
#     pip3 install torch-geometric spconv-cu$(echo ${CUDA_VERSION} | tr -d ".0") open3d

# ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0 8.6"
# ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}

# RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git
# WORKDIR /workspace/MinkowskiEngine
# RUN python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas \
# 	&& cd /workspace \
#  	&& rm -r MinkowskiEngine
	
# WORKDIR /workspace

# # Build pointops
# RUN git clone https://github.com/Pointcept/Pointcept.git
# RUN  pip install Pointcept/libs/pointops -v

# # Build pointgroup_ops
# RUN pip install Pointcept/libs/pointgroup_ops -v

# # Build swin3d
# # RUN pip install -U git+https://github.com/microsoft/Swin3D.git -v

# RUN pip3 install flash-attn --no-build-isolation

# WORKDIR /tmp
# COPY requirements.txt .
# RUN pip3 install -r requirements.txt

    

