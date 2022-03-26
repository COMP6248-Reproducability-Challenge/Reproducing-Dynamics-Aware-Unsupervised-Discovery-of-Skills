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

# Make a link to avoid an intermittent error around "-iGl not found"
# RUN ln -s /usr/src/x86_64-linux-gnu/libGL.so.1 /usr/src/x86_64-linux-gnu/libGL.so

# Install mujoco
RUN pip install -U 'mujoco-py<2.2,>=2.1'
RUN python -c "import mujoco_py"

# Install further packages
# RUN pip install --user pygame==2.1.2
# RUN pip install --user gym==0.23.1		# Just the basic gym environments for now - replace with e.g. gym[atari] for atari envs, etc
# RUN pip install torchbearer==0.5.3		# University course related package
# RUN pip install pynput==1.7.6 			# Attempting to use for demonstration learning
RUN apt-get install -y  python3-tk 	# Used for matplotlib to display plots on host machine
