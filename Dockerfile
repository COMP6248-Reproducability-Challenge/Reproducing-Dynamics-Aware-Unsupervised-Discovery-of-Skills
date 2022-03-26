FROM rocm/pytorch:rocm5.0.1_ubuntu18.04_py3.7_pytorch_staging


# Download and place the mujoco 2.1.0 software in the location required by mujoco_py:
RUN wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
RUN tar -xvzf mujoco210-linux-x86_64.tar.gz
RUN mkdir ~/.mujoco
RUN mv mujoco210 ~/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so


# Install requirements to get mujoco to be able to render on host machine from docker container:
RUN sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev
RUN sudo apt install patchelf=0.9-1


# Install mujoco
RUN pip install -U 'mujoco-py<2.2,>=2.1'
RUN python -c "import mujoco_py"


# Install further packages
RUN apt-get install -y  python3-tk 	# Used for matplotlib to display plots on host machine
