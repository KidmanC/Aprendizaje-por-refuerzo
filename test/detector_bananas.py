import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time

# Bananas relevantes están a la derecha de Kong, en cualquier altura
ROI_BANANAS = (250, 60, 960, 510)

class DetectorBananas:
    def __init__(self):
        self.titulo = "BlueStacks"
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
            return 0, None, []

        x0, y0, x1, y1 = ROI_BANANAS
        roi = frame[y0:y1, x0:x1]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        amarillo_bajo = np.array([21, 120, 150])
        amarillo_alto = np.array([32, 255, 255])

        mascara = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)
        mascara = cv2.erode(mascara, None, iterations=1)
        mascara = cv2.dilate(mascara, None, iterations=1)
        
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bananas = 0
        frame_resultado = frame.copy()
        contornos_validos = []
        
        for i, contorno in enumerate(contornos):
            area = cv2.contourArea(contorno)
            x, y, w, h = cv2.boundingRect(contorno)
            
            MIN_AREA = 40
            MAX_AREA = 200

            if not (MIN_AREA < area < MAX_AREA):
                continue
        
            ratio = w / h if h > 0 else 0
            if not (0.5 < ratio < 4):
                continue

            hull = cv2.convexHull(contorno)
            solidez = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if solidez < 0.6:
                continue
            
            # Convertir coordenadas a frame completo
            x_real = x + x0
            y_real = y + y0

            bananas += 1
            # Desplazar contorno a coordenadas del frame completo
            contorno_desplazado = contorno + np.array([x0, y0])
            contornos_validos.append(contorno_desplazado)
            cv2.rectangle(frame_resultado, (x_real, y_real), (x_real+w, y_real+h), (0, 255, 0), 2)
            cv2.putText(frame_resultado, f"{area:.0f}", (x_real, y_real-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        return bananas, frame_resultado, contornos_validos
    
    def probar(self):
        print("=== DETECTOR DE BANANAS ===")
        print("BlueStacks detectado automáticamente")
        print("Presiona 'q' para salir")
        print("Presiona 's' para guardar frame")
        time.sleep(2)
        
        cv2.namedWindow('Detector')
        
        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                print("Esperando BlueStacks...")
                time.sleep(1)
                continue
            
            cantidad, frame_con_marcas, _ = self.detectar_bananas(frame)
            
            cv2.putText(frame_con_marcas, f"Bananas: {cantidad}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Detector', frame_con_marcas)
            cv2.moveWindow('Detector', 100, 100)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"bananas_{cantidad}.png", frame_con_marcas)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DetectorBananas()
    detector.probar()