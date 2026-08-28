# syntax=docker/dockerfile:1
# Copyright (c) 2024, Robotis Lab Project Developers.
# All rights reserved.
#
# Based on Isaac Lab Dockerfile structure

# Base image from NVIDIA Isaac Sim
ARG ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION_ARG=5.1.0
FROM ${ISAACSIM_BASE_IMAGE_ARG}:${ISAACSIM_VERSION_ARG} AS base
ENV ISAACSIM_VERSION=${ISAACSIM_VERSION_ARG}

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

# Adds labels to the Dockerfile
LABEL version="1.0.0"
LABEL description="Dockerfile for building and running the Isaac Lab and robotis_lab framework inside Isaac Sim container image."

# Arguments
ARG ISAACSIM_ROOT_PATH_ARG=/isaac-sim
ENV ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH_ARG}
ARG ROBOTISLAB_PATH_ARG=/workspace/robotic_suite
ENV ROBOTISLAB_PATH=${ROBOTISLAB_PATH_ARG}

# Non-root user (UID/GID 1000) that the container will run as
ARG USERNAME=pait
ARG USER_UID=1000
ARG USER_GID=1000

ARG DOCKER_USER_HOME_ARG=/home/${USERNAME}
ENV DOCKER_USER_HOME=${DOCKER_USER_HOME_ARG}

# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    libssl-dev \
    ncurses-term \
    wget \
    curl \
    vim \
    sudo \
    python3-venv \
    python3-pip \
    python3-dev \
    libevdev-dev \
    && apt -y autoremove && apt clean autoclean \
    && rm -rf /var/lib/apt/lists/*

# ---- Create a non-root user (UID/GID 1000) ----
# The Isaac Sim base image may already ship a UID 1000 user, so rename it when
# it exists instead of failing on a duplicate UID.
RUN set -eux; \
    if getent group ${USER_GID} >/dev/null; then \
        groupmod -n ${USERNAME} "$(getent group ${USER_GID} | cut -d: -f1)"; \
    else \
        groupadd --gid ${USER_GID} ${USERNAME}; \
    fi; \
    if getent passwd ${USER_UID} >/dev/null; then \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m -g ${USER_GID} \
            "$(getent passwd ${USER_UID} | cut -d: -f1)"; \
    else \
        useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}; \
    fi; \
    if getent group render >/dev/null; then \
        usermod -aG video,render,plugdev ${USERNAME}; \
    else \
        usermod -aG video,plugdev ${USERNAME}; \
    fi; \
    if getent group isaac-sim >/dev/null; then \
        usermod -aG isaac-sim ${USERNAME}; \
    fi; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Copy Robotis Lab directories
COPY simulation/third_party/ ${ROBOTISLAB_PATH}/third_party/
COPY simulation/scripts/ ${ROBOTISLAB_PATH}/scripts/
COPY simulation/source/ ${ROBOTISLAB_PATH}/source/

# License texts and attribution must travel with the image, not just the repo:
# Apache-2.0 4(a) requires handing recipients a copy of the license, and
# BSD-3-Clause clause 2 requires reproducing the notice in binary form.
COPY LICENSE THIRD_PARTY_NOTICES.md ${ROBOTISLAB_PATH}/
COPY LICENSES/ ${ROBOTISLAB_PATH}/LICENSES/

# Fail the build here (not at runtime, 12s into an Isaac Sim startup) if the
# copied tree is incomplete. isaaclab.utils.datasets is the one that has gone
# missing before — a stray `datasets/` ignore rule eats it silently.
RUN test -f ${ROBOTISLAB_PATH}/third_party/IsaacLab/source/isaaclab/isaaclab/utils/datasets/__init__.py \
    || { echo "ERROR: isaaclab/utils/datasets/ missing from build context"; exit 1; }

# Set up Isaac Lab path from third_party
ENV ISAACLAB_PATH=${ROBOTISLAB_PATH}/third_party/IsaacLab

# Ensure isaaclab.sh has execute permissions
RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh

# Set up symbolic link for build time (will be recreated at runtime by entrypoint.sh)
RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install apt dependencies for Isaac Lab extensions
RUN ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Create directories for singularity / cache usage
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/ov && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/pip && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/nvidia/GLCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nv/ComputeCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nvidia-omniverse/logs && \
    mkdir -p ${DOCKER_USER_HOME}/.local/share/ov/data && \
    mkdir -p ${DOCKER_USER_HOME}/Documents

# Create NVIDIA binary placeholders for singularity usage
RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

#==
# Build CycloneDDS C library (required by robotis_dds_python)
#==
RUN cd ${ROBOTISLAB_PATH}/third_party/cyclonedds && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${DOCKER_USER_HOME}/cyclonedds/install \
          -DBUILD_EXAMPLES=OFF \
          -DENABLE_SECURITY=NO \
          .. && \
    cmake --build . && \
    cmake --install .

# Set CycloneDDS environment variables
ENV CYCLONEDDS_HOME=${DOCKER_USER_HOME}/cyclonedds/install
ENV CMAKE_PREFIX_PATH=${CYCLONEDDS_HOME}:${CMAKE_PREFIX_PATH}
ENV LD_LIBRARY_PATH=${CYCLONEDDS_HOME}/lib:${LD_LIBRARY_PATH}
ENV PATH=${CYCLONEDDS_HOME}/bin:${PATH}

#==
# Install all Python dependencies from pinned requirements (pip freeze)
#==
COPY requirements.txt /tmp/requirements.txt
# flatdict==4.0.1 uses legacy setup.py that needs pkg_resources; pre-install it before main install
# so pip skips rebuilding it during the main pass (which uses normal isolation for rl_games/poetry)
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install "setuptools>=65.0" && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-build-isolation "flatdict==4.0.1" && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-deps -r /tmp/requirements.txt

# Create a separate Python virtual environment for LeRobot
ENV LEROBOT_VENV=${DOCKER_USER_HOME}/lerobot_env
RUN python3 -m venv ${LEROBOT_VENV}

# Install LeRobot from PyPI
RUN ${LEROBOT_VENV}/bin/python3 -m pip install --upgrade pip && \
    ${LEROBOT_VENV}/bin/python3 -m pip install lerobot==0.3.3 && \
    ${LEROBOT_VENV}/bin/python3 -m pip install h5py

# Set up aliases and environment
RUN echo "# Environment variables" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "export ROBOTISLAB_PATH=${ROBOTISLAB_PATH}" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "export CYCLONEDDS_HOME=${CYCLONEDDS_HOME}" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "export CMAKE_PREFIX_PATH=${CYCLONEDDS_HOME}:\$CMAKE_PREFIX_PATH" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "export LD_LIBRARY_PATH=${CYCLONEDDS_HOME}/lib:\$LD_LIBRARY_PATH" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "export PATH=${CYCLONEDDS_HOME}/bin:\$PATH" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "export ROS_DOMAIN_ID=30" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "export LEROBOT_VENV=${LEROBOT_VENV}" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "# Bash settings" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "shopt -s histappend" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "PROMPT_COMMAND='history -a'" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "# Isaac Lab aliases" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias pip3='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "# LeRobot venv aliases" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias lerobot-activate='source \${LEROBOT_VENV}/bin/activate'" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias lerobot-python='\${LEROBOT_VENV}/bin/python3'" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias lerobot-pip='\${LEROBOT_VENV}/bin/pip'" >> ${DOCKER_USER_HOME}/.bashrc

# Create real wrapper scripts so these work as docker run commands (not just bash aliases)
RUN printf '#!/bin/bash\nexec ${LEROBOT_VENV}/bin/python3 "$@"\n' > /usr/local/bin/lerobot-python && \
    chmod +x /usr/local/bin/lerobot-python && \
    printf '#!/bin/bash\nexec ${LEROBOT_VENV}/bin/pip "$@"\n' > /usr/local/bin/lerobot-pip && \
    chmod +x /usr/local/bin/lerobot-pip

# Hand ownership of the home directory, workspace, and Isaac Sim cache to the
# non-root user so it can read/write logs, caches, and mounted volumes at runtime.
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/data ${ISAACSIM_ROOT_PATH}/kit/logs && \
    chown -R ${USER_UID}:${USER_GID} ${DOCKER_USER_HOME} ${ROBOTISLAB_PATH} \
    ${ISAACSIM_ROOT_PATH}/kit/cache ${ISAACSIM_ROOT_PATH}/kit/data ${ISAACSIM_ROOT_PATH}/kit/logs

USER ${USERNAME}
WORKDIR ${ROBOTISLAB_PATH}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
