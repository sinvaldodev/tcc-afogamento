import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

def processar_video(caminho_entrada, caminho_saida, modelo, pontos_piscina):
    """
    Processa um único vídeo frame a frame, aplicando a detecção do YOLO e a validação de ROI.
    """
    cap = cv2.VideoCapture(caminho_entrada)
    if not cap.isOpened():
        print(f"  [ERRO] Não foi possível abrir o vídeo: {caminho_entrada}")
        return

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(caminho_saida, fourcc, fps, (largura, altura))

    print(f"  Iniciando rastreamento de frames...")
    
    while True:
        sucesso, frame = cap.read()
        if not sucesso:
            break

        # Desenha a área da piscina no frame em azul claro para visualização
        cv2.polylines(frame, [pontos_piscina], isClosed=True, color=(255, 255, 0), thickness=2)
        
        # Usando a primeira coordenada da ROI para posicionar o texto de forma dinâmica
        pos_texto_roi = (pontos_piscina[0][0][0], pontos_piscina[0][0][1] - 10)
        cv2.putText(frame, 'ROI (Piscina)', pos_texto_roi, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Roda a IA focando apenas em pessoas (classes=0) e ajustando para vídeos com ruído (imgsz=640)
        resultados = modelo.predict(frame, classes=0, conf=0.25, imgsz=640, verbose=False)
        
        # Pega as coordenadas de cada pessoa detectada
        for caixa in resultados[0].boxes:
            # Coordenadas do retângulo da pessoa
            x1, y1, x2, y2 = map(int, caixa.xyxy[0])
            
            # Calcula o centro geométrico da pessoa (centro de massa)
            centro_x = int((x1 + x2) / 2)
            centro_y = int((y1 + y2) / 2)

            # Validação matemática: o centro da pessoa está dentro do polígono da piscina?
            esta_na_agua = cv2.pointPolygonTest(pontos_piscina, (centro_x, centro_y), False) >= 0

            if esta_na_agua:
                # Pessoa na piscina -> Caixa Vermelha
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (centro_x, centro_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, 'NA AGUA', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Pessoa fora da piscina -> Caixa Verde
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (centro_x, centro_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, 'BORDA', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"  [CONCLUÍDO] Salvo como: {caminho_saida}\n")


if __name__ == "__main__":
    # Pega o caminho absoluto de onde este script (processar_roi.py) está salvo
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    
    # Garante que a pasta dataset seja buscada exatamente no mesmo local do script
    pasta_dataset = os.path.join(diretorio_do_script, 'dataset')
    
    # 1. PREPARAÇÃO (Executado apenas uma vez)
    print("Carregando motor YOLOv8 na memória...")
    modelo_yolo = YOLO('yolov8s.pt')
    
    # Definindo a ROI
    roi_piscina = np.array([[100, 200], [500, 200], [600, 500], [50, 500]], np.int32)
    roi_piscina = roi_piscina.reshape((-1, 1, 2))

    # 2. MAPEAMENTO DOS ARQUIVOS (.mp4)
    caminhos_videos = glob.glob(os.path.join(pasta_dataset, '*.mp4'))
    
    # 3. FILA DE PROCESSAMENTO
    if not caminhos_videos:
        print(f"[AVISO] Nenhum vídeo (.mp4) encontrado em: {pasta_dataset}")
    else:
        print(f"\nEncontrados {len(caminhos_videos)} vídeos. Iniciando processamento em lote...\n")
        
        for caminho in caminhos_videos:
            nome_arquivo = os.path.basename(caminho)
            
            # Salva o resultado no mesmo diretório do script para evitar sumiço de arquivos
            caminho_saida = os.path.join(diretorio_do_script, f"lote_resultado_{nome_arquivo}")
            
            print(f"-> Analisando: {nome_arquivo}")
            processar_video(caminho, caminho_saida, modelo_yolo, roi_piscina)
            
        print("Processamento em lote finalizado com sucesso!")