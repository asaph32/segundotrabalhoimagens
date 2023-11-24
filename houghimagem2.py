import cv2
import numpy as np

# Carregando a imagem
image = cv2.imread('imagem2.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicando suavização para reduzir ruído
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Aplicando a transformada de Hough para círculos
circles = cv2.HoughCircles(image_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=30)

# Desenhando os círculos detectados na imagem original
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

# Exibindo a imagem original e a imagem com círculos detectados
cv2.imshow('Imagem Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()