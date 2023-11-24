import cv2

# Carregar a imagem em escala de cinza
image = cv2.imread('imagem1.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar a limiarização adaptativa usando o método de Gaussiano
# Parâmetros: (imagem, valor máximo, método de limiarização, tipo de limiarização,
# tamanho do bloco, constante a ser subtraída do valor médio)
adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

# Exibir a imagem original e a imagem com limiarização adaptativa
cv2.imshow('Original', image)
cv2.imshow('Limiarização Adaptativa', adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()