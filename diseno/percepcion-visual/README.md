# Percepción Visual para Bot Autónomo en Videojuegos

## Explicación del tema

Este submódulo es responsable de **interpretar visualmente el estado del juego** a partir de la captura de pantalla. Mientras el módulo de captura se encarga de obtener la imagen, la percepción visual analiza esa imagen para extraer información estructurada que el agente de aprendizaje por refuerzo pueda entender.

### ¿Qué problema resuelve?
Los videojuegos son cajas negras: no tenemos acceso a sus variables internas (posición del personaje, vida restante, enemigos presentes). La percepción visual actúa como los "ojos" del bot, transformando píxeles en información útil.

### ¿Qué información extraemos?
- **HUD (Head-Up Display):** Vida, puntaje, tiempo restante, munición
- **Elementos del juego:** Posición del personaje, enemigos, obstáculos, objetos recolectables
- **Eventos visuales:** Destellos, cambios de color, animaciones que indican peligro o recompensa
- **Estado del nivel:** ¿El personaje está en el aire? ¿Hay una puerta abierta?

### Importancia para el Grupo 2 (Aprendizaje por Refuerzo)
El agente RL necesita una representación del estado del juego para tomar decisiones. Esta representación puede ser:
- **Directa:** La imagen redimensionada (píxeles crudos) como entrada a una red convolucional
- **Estructurada:** Variables extraídas (ej. "enemigo_cerca = True", "vida = 45") como entrada a una red densa

Ambos enfoques son válidos y exploraremos ambos en este módulo.

---

## Enlaces relevantes

### Captura de pantalla (base necesaria)
- [MSS - captura rápida en Python](https://github.com/BoboTiG/python-mss)
- [DXcam - captura de alta velocidad (Windows)](https://github.com/ra1nty/DXcam)
- [OpenCV - procesamiento de imágenes](https://docs.opencv.org/)

### Visión por Computador clásica
- [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/)
- [Template Matching en OpenCV](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- [Detección de colores HSV](https://realpython.com/python-opencv-color-spaces/)

### Deep Learning para Visión
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) - Detección de objetos en tiempo real
- [YOLOv8 Docs - detección personalizada](https://docs.ultralytics.com/tasks/detect/)
- [PyTorch - redes convolucionales](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

### OCR (Lectura de texto)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [pytesseract - wrapper Python](https://pypi.org/project/pytesseract/)
- [EasyOCR - alternativa moderna](https://github.com/JaidedAI/EasyOCR)

### Tutoriales relacionados
- [Building a bot that plays games (serie)](https://www.youtube.com/watch?v=ZZY9YE5rZJg)
- [OpenCV para videojuegos](https://www.learncodebygaming.com/blog/tag/opencv)

---

## Códigos de prueba

### 1. Estructura base: captura + preprocesamiento
```python
import cv2
import numpy as np
from mss import mss

class PerceptorVisual:
    def __init__(self, region=(0, 0, 800, 600)):
        self.sct = mss()
        self.region = {"top": region[1], "left": region[0], 
                       "width": region[2], "height": region[3]}
    
    def capturar(self):
        """Captura la pantalla y devuelve imagen BGR"""
        img = np.array(self.sct.grab(self.region))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def preprocesar(self, img):
        """Escala de grises, suavizado, etc."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

# Uso básico
perceptor = PerceptorVisual()
while True:
    img = perceptor.capturar()
    procesada = perceptor.preprocesar(img)
    cv2.imshow("Preprocesada", procesada)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

### 2. Detección de vida/barra de progreso
```python
def detectar_barra_vida(img, region_vida=(100, 50, 200, 30)):
    """
    Detecta el porcentaje de una barra de vida horizontal
    region_vida: (x, y, ancho, alto) donde buscar
    """
    x, y, w, h = region_vida
    recorte = img[y:y+h, x:x+w]
    
    # Convertir a HSV y filtrar color rojo (típico de vida baja)
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    # Rango rojo (puede variar según el juego)
    mascara = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    
    # Calcular porcentaje de píxeles rojos
    pixeles_rojos = cv2.countNonZero(mascara)
    porcentaje_vida = (pixeles_rojos / (w * h)) * 100
    
    return porcentaje_vida

# Uso
img = perceptor.capturar()
vida = detectar_barra_vida(img, region_vida=(50, 30, 150, 15))
print(f"Vida actual: {vida:.1f}%")
```