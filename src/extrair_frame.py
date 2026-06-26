import cv2
import os

def capturar_frame_referencia():
    # Caminho dinâmico para evitar aquele problema do Docker
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    # Coloque aqui o nome do vídeo que você quer usar como base
    caminho_video = os.path.join(diretorio_atual, 'dataset', 'video1.mp4') 
    caminho_imagem = os.path.join(diretorio_atual, 'referencia_piscina.jpg')

    cap = cv2.VideoCapture(caminho_video)
    
    if not cap.isOpened():
        print(f"Erro: Não encontrei o vídeo {caminho_video}")
        return

    # Lê apenas o primeiro frame
    sucesso, frame = cap.read()
    
    if sucesso:
        cv2.imwrite(caminho_imagem, frame)
        print(f"Sucesso! Imagem salva em: {caminho_imagem}")
    else:
        print("Erro ao ler o frame do vídeo.")
        
    cap.release()

if __name__ == "__main__":
    capturar_frame_referencia()