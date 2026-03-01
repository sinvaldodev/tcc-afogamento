import torch
from ultralytics import YOLO

def validar_ia():
    print(f"--- Validando Ambiente de IA ---")
    print(f"PyTorch detectado: {torch.__version__}")
    
    # Verifica se há GPU (no seu caso, deve dar False agora, o que é normal)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executando em: {dispositivo}")

    # Tenta carregar o modelo YOLOv8 (ele vai baixar o arquivo .pt automaticamente na primeira vez)
    print("Carregando modelo de detecção...")
    modelo = YOLO('yolov8n.pt') 
    
    print("\n[SUCESSO] O ambiente está pronto para o seu TCC!")

if __name__ == "__main__":
    validar_ia()
