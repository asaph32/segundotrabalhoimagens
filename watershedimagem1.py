import cv2
import numpy as np

# Carregando a imagem
image = cv2.imread('imagem1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicando a detecção de bordas (Sobel, neste exemplo)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Convertendo a magnitude para uint8
magnitude = np.uint8(magnitude)

# Aplicando limiarização para criar uma máscara binária
_, thresh = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Aplicando a transformada de Watershed
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations=3)
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # Linhas de watershed marcadas em vermelho

# Exibindo a imagem original e a imagem segmentada com Watershed
cv2.imshow('Imagem Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()