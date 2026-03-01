import cv2
import os

def simular_monitoramento(caminho_entrada, caminho_saida):
    # Inicializa o leitor de vídeo
    cap = cv2.VideoCapture(caminho_entrada)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo '{caminho_entrada}'. Verifique se o arquivo existe.")
        return

    # Extrai as propriedades do vídeo original para criar o vídeo de saída com a mesma qualidade
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Iniciando: {largura}x{altura} a {fps} FPS. Total: {total_frames} frames.")

    # Configura o gravador de vídeo (o codec mp4v funciona muito bem nativamente)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(caminho_saida, fourcc, fps, (largura, altura))

    frames_processados = 0

    # Loop principal: lê e processa o vídeo quadro a quadro
    while True:
        sucesso, frame = cap.read()
        
        # Se 'sucesso' for False, o vídeo acabou
        if not sucesso:
            break 

        # --- ÁREA DE PROCESSAMENTO (Sua IA entrará aqui no futuro) ---
        
        # Vamos desenhar um retângulo simulando a área útil da piscina
        margem = 50
        cv2.rectangle(frame, (margem, margem), (largura - margem, altura - margem), (255, 0, 0), 2)
        
        # Adicionando um texto de status simulando a detecção
        cv2.putText(frame, 'Status: Monitoramento Ativo', (margem + 10, margem + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Contador de frames
        cv2.putText(frame, f'Frame: {frames_processados}/{total_frames}', (margem + 10, margem + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # -------------------------------------------------------------

        # Grava o frame alterado no novo arquivo
        out.write(frame)
        frames_processados += 1

        # Feedback visual no terminal a cada 50 frames processados
        if frames_processados % 50 == 0:
            print(f"Progresso: {frames_processados}/{total_frames} frames processados...")

    # Boa prática: liberar os recursos da memória ao finalizar
    cap.release()
    out.release()
    print(f"\nProcessamento concluído! Abra o arquivo '{caminho_saida}' para ver o resultado.")

if __name__ == "__main__":
    video_original = 'piscina.mp4' 
    video_processado = 'resultado_monitoramento.mp4'
    
    # Validação rápida de diretório
    if os.path.exists(video_original):
        simular_monitoramento(video_original, video_processado)
    else:
        print(f"Atenção: Coloque um arquivo de vídeo chamado '{video_original}' na mesma pasta deste script.")