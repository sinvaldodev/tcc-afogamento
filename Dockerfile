FROM python:3.10-slim

WORKDIR /app

# Atualiza o pip
RUN pip install --no-cache-dir --upgrade pip

# Instala todas as bibliotecas do requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove as versões erradas e reinstala o headless, TRAVANDO o numpy na versão 1.x
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless==4.9.0.80 numpy==1.26.4

CMD ["tail", "-f", "/dev/null"]