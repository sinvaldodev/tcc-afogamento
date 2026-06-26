import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
import json
import math # Necessário para o cálculo da distância Euclidiana

def carregar_roi_json(caminho_arquivo):
    """
    Lê o arquivo JSON e converte para a matriz NumPy exigida pelo OpenCV.
    """
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            pontos = np.array(dados["coordenadas"], np.int32)
            return pontos.reshape((-1, 1, 2)), dados.get("nome_cenario", "Cenário Desconhecido")
    except Exception as e:
        print(f"  [ERRO] Falha ao ler o arquivo JSON: {e}")
        return None, None

def processar_video(caminho_entrada, caminho_saida, modelo, pontos_piscina, nome_roi):
    """
    Processa o vídeo aplicando YOLOv8 em modo Tracking e validação de submersão
    com base em heurística espacial (Reidentificação).
    """
    cap = cv2.VideoCapture(caminho_entrada)
    if not cap.isOpened():
        print(f"\n  [ERRO] Não foi possível abrir o vídeo: {caminho_entrada}")
        return

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(caminho_saida, fourcc, fps, (largura, altura))

    # --- VARIÁVEIS DE ESTADO E CONFIGURAÇÕES DO RASTREAMENTO ---
    pessoas_na_agua_ultimo_frame = {} # Estrutura: {id_pessoa: (centro_x, centro_y)}
    pessoas_submersas = {}            # Estrutura: {id_pessoa: {'frames': 0, 'ultima_pos': (x, y)}}
    
    LIMITE_FRAMES_SUBMERSO = fps * 3  # Alerta crítico após 3 segundos de desaparecimento na água
    RAIO_TOLERANCIA = 80              # Distância máxima em pixels para fundir um "novo" ID a um ID perdido

    print(f"\n  Processando frames com Algoritmo de Re-ID... Aguarde.")
    
    while True:
        sucesso, frame = cap.read()
        if not sucesso:
            break

        # Desenha a geometria fixa da piscina no frame
        cv2.polylines(frame, [pontos_piscina], isClosed=True, color=(255, 255, 0), thickness=2)
        pos_texto_roi = (pontos_piscina[0][0][0], pontos_piscina[0][0][1] - 10)
        cv2.putText(frame, f'ROI: {nome_roi}', pos_texto_roi, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 1. IA EM MODO TRACKING (persist=True mantém os IDs consistentes entre frames)
        resultados = modelo.track(frame, persist=True, classes=0, conf=0.25, imgsz=640, verbose=False)
        
        pessoas_na_agua_agora = {} # Guardará os IDs ativos no frame atual: {id: (centro_x, centro_y)}

        if resultados[0].boxes.id is not None:
            ids = resultados[0].boxes.id.int().tolist()
            caixas = resultados[0].boxes.xyxy.int().tolist()

            for id_pessoa, caixa in zip(ids, caixas):
                x1, y1, x2, y2 = caixa
                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)

                # Teste geométrico corrigido: o centro de massa está dentro da piscina?
                esta_na_agua = cv2.pointPolygonTest(pontos_piscina, (centro_x, centro_y), False) >= 0

                if esta_na_agua:
                    id_real = id_pessoa 
                    
                    # --- FILTRO DE REIDENTIFICAÇÃO ESPACIAL ---
                    # Se este ID acabou de ser criado pela IA, verifica se não está muito próximo de alguém que afundou
                    for id_perdido, dados in list(pessoas_submersas.items()):
                        pos_antiga = dados['ultima_pos']
                        
                        # Distância Euclidiana
                        distancia = math.sqrt((centro_x - pos_antiga[0])**2 + (centro_y - pos_antiga[1])**2)
                        
                        if distancia < RAIO_TOLERANCIA:
                            # Reidentificado! Recupera o ID original e descarta o novo gerado pela "piscada" da IA
                            id_real = id_perdido
                            del pessoas_submersas[id_perdido]
                            break 

                    pessoas_na_agua_agora[id_real] = (centro_x, centro_y)

                    # Desenha a marcação visual de segurança na superfície
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (centro_x, centro_y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f'ID {id_real}: Superficie', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Alvo fora da água (Borda)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(frame, f'ID {id_pessoa}: Borda', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 2. DETECÇÃO DE DESAPARECIMENTO DENTRO DA ÁGUA
        # Se estava mapeado na água no frame anterior, mas sumiu no atual, vai para a lista de suspeitos
        for id_antigo, pos_antiga in pessoas_na_agua_ultimo_frame.items():
            if id_antigo not in pessoas_na_agua_agora:
                if id_antigo not in pessoas_submersas:
                    # Inicia cronômetro armazenando a última coordenada espacial válida
                    pessoas_submersas[id_antigo] = {'frames': 0, 'ultima_pos': pos_antiga}

        # 3. MÁQUINA DE ESTADOS: ATUALIZAÇÃO DOS CRONÔMETROS DE ALERTA
        y_alerta = 50 
        for id_submerso in list(pessoas_submersas.keys()):
            pessoas_submersas[id_submerso]['frames'] += 1
            
            # Valida se ultrapassou o tempo seguro estipulado
            if pessoas_submersas[id_submerso]['frames'] > LIMITE_FRAMES_SUBMERSO:
                cv2.putText(frame, f"ALERTA CRITICO: ID {id_submerso} POSSIVELMENTE SUBMERSO!", 
                            (50, y_alerta), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                y_alerta += 40 

        # Atualiza o estado para a comparação do próximo frame
        pessoas_na_agua_ultimo_frame = pessoas_na_agua_agora.copy()

        out.write(frame)

    cap.release()
    out.release()
    print(f"  [CONCLUÍDO] Vídeo de experimento salvo em: {caminho_saida}\n")


if __name__ == "__main__":
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    pasta_dataset = os.path.join(diretorio_do_script, 'dataset')
    pasta_rois = os.path.join(diretorio_do_script, 'rois')

    os.makedirs(pasta_rois, exist_ok=True)

    caminhos_videos = sorted(glob.glob(os.path.join(pasta_dataset, '*.mp4')))
    caminhos_rois = sorted(glob.glob(os.path.join(pasta_rois, '*.json')))
    
    if not caminhos_videos:
        print(f"[ERRO] Nenhum vídeo .mp4 encontrado em: {pasta_dataset}")
        exit()
        
    if not caminhos_rois:
        print(f"[ERRO] Nenhum arquivo JSON de ROI encontrado em: {pasta_rois}")
        print("Adicione um arquivo de configuração (.json) mapeando as coordenadas antes de executar.")
        exit()

    print("\n=======================================================")
    print("      PAINEL DE EXPERIMENTOS - VISÃO COMPUTACIONAL     ")
    print("=======================================================")
    
    # --- INTERFACE DE SELEÇÃO: VÍDEO ---
    print("\n[PASSO 1] Selecione o vídeo para o teste:")
    for indice, caminho in enumerate(caminhos_videos):
        print(f"  [{indice + 1}] {os.path.basename(caminho)}")
    
    while True:
        try:
            escolha_video = int(input(f"Selecione o número do vídeo (1 a {len(caminhos_videos)}): "))
            if 1 <= escolha_video <= len(caminhos_videos):
                video_selecionado = caminhos_videos[escolha_video - 1]
                break
            print("Opção inválida.")
        except ValueError:
            print("Por favor, introduza apenas números inteiros.")

    # --- INTERFACE DE SELEÇÃO: CONFIGURAÇÃO ROI ---
    print("\n[PASSO 2] Selecione o arquivo de calibração geométrica (JSON):")
    for indice, caminho in enumerate(caminhos_rois):
        print(f"  [{indice + 1}] {os.path.basename(caminho)}")
        
    while True:
        try:
            escolha_roi = int(input(f"Selecione o número da ROI (1 a {len(caminhos_rois)}): "))
            if 1 <= escolha_roi <= len(caminhos_rois):
                roi_selecionada = caminhos_rois[escolha_roi - 1]
                break
            print("Opção inválida.")
        except ValueError:
            print("Por favor, introduza apenas números inteiros.")

    # --- INICIALIZAÇÃO DA INFERÊNCIA ---
    print("\n=======================================================")
    print("A carregar parâmetros de configuração...")
    matriz_pontos, nome_cenario = carregar_roi_json(roi_selecionada)
    
    if matriz_pontos is None:
        exit()

    print("A carregar modelo YOLOv8s em memória...")
    modelo_yolo = YOLO('yolov8s.pt')

    nome_arquivo_saida = f"experimento_{os.path.basename(video_selecionado)}"
    caminho_saida = os.path.join(diretorio_do_script, nome_arquivo_saida)
    
    print(f"\n-> Configuração de Execução:")
    print(f"   Mídia Alvo: {os.path.basename(video_selecionado)}")
    print(f"   Malha de ROI Aplicada: {nome_cenario}")
    print("=======================================================")
    
    processar_video(video_selecionado, caminho_saida, modelo_yolo, matriz_pontos, nome_cenario)