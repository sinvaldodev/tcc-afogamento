import cv2
import numpy as np

def testar_ambiente():
    print(f"Versão do OpenCV: {cv2.__version__}")
    print(f"Versão do Numpy: {np.__version__}")
    
    # Criando uma imagem "falsa" (um frame preto de 800x600) para simular o feed de uma câmera
    frame_simulado = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Adicionando um texto na imagem
    cv2.putText(frame_simulado, 'Ambiente TCC - Deteccao de Afogamento OK!', 
                (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print("Processamento de matrizes (Numpy) e imagens (OpenCV) funcionando perfeitamente!")

if __name__ == "__main__":
    testar_ambiente()