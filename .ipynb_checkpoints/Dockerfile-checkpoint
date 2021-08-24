# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM python:3.7
# Create working directory
# Copy contents
WORKDIR /app
COPY . /app

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

RUN sudo apt-get remove --purge ffmpeg
RUN sudo apt-add-repository ppa:mc3man/trusty-media
RUN sudo apt-add-repository ppa:jonathonf/ffmpeg-3
RUN sudo apt-get update
RUN sudo apt-get install ffmpeg

RUN pip install streamlit-drawable-canvas
RUN pip install detecto
#Install dependencies (pip or conda)
RUN pip install -r requirements.txt


CMD ["python", "-m", "streamlit.cli", "run", "stvideo.py", "--server.port=8888"]
EXPOSE 8888