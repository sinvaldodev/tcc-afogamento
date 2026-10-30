from ultralytics import YOLO

# 1. Carrega o modelo base pré-treinado (recomenda-se o Nano por ser mais leve)
model = YOLO('yolov8n.pt')

# 2. Inicia o treinamento usando o arquivo data.yaml do Roboflow
# Ajuste o 'epochs' dependendo da força do seu processador/placa de vídeo
print("Iniciando treinamento do modelo...")
results = model.train(
    data='./tcc-afogamento/src/dataset/Swimming-Drowning.v5i.yolov8/data.yaml', 
    epochs=50,       # Número de vezes que a IA vai ver o dataset inteiro
    imgsz=640,       # Tamanho da imagem (padrão do YOLO)
    batch=16,        # Quantidade de imagens processadas por vez
    name='modelo_tcc_v1' # Nome da pasta onde os resultados serão salvos
)