import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time

class DetectorBananas:
    def __init__(self):
        self.titulo = "BlueStacks"  # Busca por este título
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()
    
    def actualizar_ventana(self):
        """Busca la ventana de BlueStacks y actualiza coordenadas"""
        ventanas = gw.getWindowsWithTitle(self.titulo)
        if ventanas:
            self.ventana = ventanas[0]
            self.monitor = {
                "top": self.ventana.top,
                "left": self.ventana.left,
                "width": self.ventana.width,
                "height": self.ventana.height
            }
            return True
        return False
    
    def capturar_pantalla(self):
        """Captura la ventana de BlueStacks sin importar dónde esté"""
        if not self.actualizar_ventana():
            print("No encuentro BlueStacks")
            return None
        
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    def detectar_bananas(self, frame):
        if frame is None:
            return 0, None
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        ''' Por esto (banana kong amarillo puro):
        amarillo_bajo = np.array([18, 120, 150])
        amarillo_alto = np.array([32, 255, 255])
        '''
        
        amarillo_bajo = np.array([21, 120, 150])
        amarillo_alto = np.array([32, 255, 255])

        mascara = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)
        
        # Sin filtros extra, solo limpieza básica
        mascara = cv2.erode(mascara, None, iterations=1)
        mascara = cv2.dilate(mascara, None, iterations=1)
        
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bananas = 0
        frame_resultado = frame.copy()
        
        # Dibujar TODOS los contornos sin filtrar por área
        for i, contorno in enumerate(contornos):
            area = cv2.contourArea(contorno)
            x, y, w, h = cv2.boundingRect(contorno)
            
            # Después (ajusta según resolución de BlueStacks):
            MIN_AREA = 40   # Banana pequeña ~100-800 px²
            MAX_AREA = 200   # Banana grande ~3000-6000 px²

            if not (MIN_AREA < area < MAX_AREA):
                continue
        
            x, y, w, h = cv2.boundingRect(contorno)
            ratio = w / h if h > 0 else 0

            # Las bananas en Banana Kong tienen ratio ~1.8 a 4.0
            if not (0.5 < ratio < 4):
                continue

            
            # Bonus: también puedes filtrar por solidez
            hull = cv2.convexHull(contorno)
            solidez = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if solidez < 0.6:   # Forma muy irregular → no es banana
                continue
            
            bananas += 1
            cv2.rectangle(frame_resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame_resultado, f"{area:.0f}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        # Mostrar máscara para depuración
        #cv2.imshow('Mascara', mascara)
        #cv2.moveWindow('Mascara', 800, 100)
        
        return bananas, frame_resultado
    
    def probar(self):
        print("=== DETECTOR DE BANANAS ===")
        print("BlueStacks detectado automáticamente")
        print("Puedes mover BlueStacks donde quieras")
        print("Presiona 'q' para salir")
        print("Presiona 's' para guardar frame")
        time.sleep(2)
        
        cv2.namedWindow('Detector')
        
        while True:
            # Capturar (siempre busca la ventana actualizada)
            frame = self.capturar_pantalla()
            if frame is None:
                print("Esperando BlueStacks...")
                time.sleep(1)
                continue
            
            # Detectar bananas
            cantidad, frame_con_marcas = self.detectar_bananas(frame)
            
            # Mostrar info
            cv2.putText(frame_con_marcas, f"Bananas: {cantidad}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Detector', frame_con_marcas)
            cv2.moveWindow('Detector', 100, 100)  # Posición fija para la ventana de captura
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"bananas_{cantidad}.png", frame_con_marcas)
        
        cv2.destroyAllWindows()

# Ejecutar
detector = DetectorBananas()
detector.probar()