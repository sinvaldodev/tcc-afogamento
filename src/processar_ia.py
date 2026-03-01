import cv2
from ultralytics import YOLO
import os

def processar_video_ia(caminho_entrada, caminho_saida):
    print(f"\n--- Iniciando análise de IA no {caminho_entrada} ---")
    
    # Carrega o modelo YOLOv8 na versão 'nano' (n), que é mais leve para rodar apenas com CPU
    print("Carregando modelo YOLOv8...")
    modelo = YOLO('yolov8s.pt')

    cap = cv2.VideoCapture(caminho_entrada)
    if not cap.isOpened():
        print(f"Erro: Não encontrei o vídeo '{caminho_entrada}'. Ele está na pasta src/?")
        return

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configuração do vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(caminho_saida, fourcc, fps, (largura, altura))

    frames_processados = 0

    while True:
        sucesso, frame = cap.read()
        if not sucesso:
            break

        # A MÁGICA ACONTECE AQUI:
        # classes=0 diz para o YOLO ignorar carros, cachorros, etc., e focar SÓ em pessoas.
        # conf=0.3 diz para só marcar se a IA tiver pelo menos 30% de certeza.
        resultados = modelo.predict(frame, classes=0, conf=0.15, imgsz=640, verbose=False)

        # O YOLO desenha os retângulos automaticamente no frame
        frame_anotado = resultados[0].plot()

        out.write(frame_anotado)
        frames_processados += 1

        if frames_processados % 30 == 0:
            print(f"Progresso: {frames_processados}/{total_frames} frames processados...")

    cap.release()
    out.release()
    print(f"Análise concluída! O resultado foi salvo como: {caminho_saida}")


if __name__ == "__main__":
    # Lista com os nomes dos seus vídeos de teste
    videos_para_testar = ['video1.mp4', 'video2.mp4']
    
    for video in videos_para_testar:
        if os.path.exists(video):
            nome_saida = f"ia_resultado_{video}"
            processar_video_ia(video, nome_saida)
        else:
            print(f"\nAviso: O arquivo '{video}' não foi encontrado na pasta atual.")