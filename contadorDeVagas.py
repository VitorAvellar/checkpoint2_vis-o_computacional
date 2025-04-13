import cv2
import numpy as np

# Coordenadas das áreas de interesse (simulando vagas de estacionamento)
area1 = [1, 89, 108, 213]
area2 = [115, 87, 152, 211]
area3 = [289, 89, 138, 212]
area4 = [439, 87, 135, 212]
area5 = [591, 90, 132, 206]
area6 = [738, 93, 139, 204]
area7 = [881, 93, 138, 201]
area8 = [1027, 94, 147, 202]

areasMonitoradas = [area1, area2, area3, area4, area5, area6, area7, area8]

# Abre o vídeo
captura = cv2.VideoCapture('video.mp4')

while True:
    sucesso, frame = captura.read()
    # Converte a imagem para escala de cinza
    escalaCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplica threshold adaptativo para destacar objetos
    imagemThreshold = cv2.adaptiveThreshold(escalaCinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    # Suaviza a imagem para reduzir ruídos
    imagemSuavizada = cv2.medianBlur(imagemThreshold, 5)
    # Dilata a imagem para destacar os contornos
    estrutura = np.ones((3, 3), np.int8)
    imagemDilatada = cv2.dilate(imagemSuavizada, estrutura)

    contagemLivres = 0
    for posX, posY, largura, altura in areasMonitoradas:
        # Recorta a área monitorada da imagem processada
        recorteArea = imagemDilatada[posY:posY+altura, posX:posX+largura]
        # Conta os pixels brancos (presença de carro = mais pixels brancos)
        pixelsBrancos = cv2.countNonZero(recorteArea)

        # Mostra o número de pixels brancos sobre a vaga
        cv2.putText(frame, str(pixelsBrancos), (posX, posY+altura-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Define a cor do retângulo dependendo da presença de carro
        if pixelsBrancos > 3000:
            cv2.rectangle(frame, (posX, posY), (posX + largura, posY + altura), (0, 0, 255), 3)  # Ocupada (vermelho)
        else:
            cv2.rectangle(frame, (posX, posY), (posX + largura, posY + altura), (0, 255, 0), 3)  # Livre (verde)
            contagemLivres += 1

    # Mostra o total de vagas livres na tela
    cv2.rectangle(frame, (90, 0), (415, 60), (255, 0, 0), -1)
    cv2.putText(frame, f'LIVRE: {contagemLivres}/8', (95, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    # Exibe as imagens
    cv2.imshow('Video Original', frame)
    cv2.imshow('Processamento', imagemDilatada)
    cv2.waitKey(1)
