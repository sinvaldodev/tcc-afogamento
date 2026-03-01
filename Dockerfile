FROM python:3.10-slim

WORKDIR /app

# Copiar e instalar as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Manter o container rodando para desenvolvimento
CMD ["tail", "-f", "/dev/null"]
