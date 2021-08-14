# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM python:3.7
# Create working directory
# Copy contents
WORKDIR /app
COPY . /app
RUN apt-get update
RUN apt install -y libgl1-mesa-glx


RUN pip install streamlit-drawable-canvas
RUN pip install detecto
#Install dependencies (pip or conda)
RUN pip install -r requirements.txt

EXPOSE 8083
ENTRYPOINT ["streamlit","run"]
CMD ["stvideo1.py"]