"""
detector_gameover.py — Detección de pantalla Revive/Game Over en Banana Kong
Resolución BlueStacks: 960x540

Estrategia: Template matching del cartel "Revive?" que aparece al morir.
Se usa solo la zona superior del cartel (texto "Revive?") para ignorar
el número de vidas restantes que cambia entre partidas.
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import os


# ── Configuración ────────────────────────────────────────────────────

# Umbral de confianza para aceptar que es la pantalla de Revive
UMBRAL = 0.60

# Escalas a probar
ESCALAS = [0.9, 1.0, 1.1]

# El cartel aparece centrado en pantalla — buscamos en la zona central
# (left, top, right, bottom) en píxeles para 960x540
ROI = (200, 100, 760, 400)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


class DetectorGameOver:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()

        # Cargar template del texto "Revive?" (sin el número)
        ruta = os.path.join(TEMPLATES_DIR, "revive_texto.png")
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se encontró el template: {ruta}")
        self.template = img
        print(f"✅ Template revive cargado: {img.shape[1]}x{img.shape[0]}px")

    # ─────────────────────────────────────────
    def actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle(self.titulo)
        if ventanas:
            self.ventana = ventanas[0]
            self.monitor = {
                "top": self.ventana.top,
                "left": self.ventana.left,
                "width": self.ventana.width,
                "height": self.ventana.height,
            }
            return True
        return False

    def capturar_pantalla(self):
        if self.ventana is None:
            if not self.actualizar_ventana():
                return None
        try:
            self.monitor = {
                "top": self.ventana.top,
                "left": self.ventana.left,
                "width": self.ventana.width,
                "height": self.ventana.height,
            }
        except Exception:
            self.ventana = None
            return None
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ─────────────────────────────────────────
    def detectar_gameover(self, frame):
        """
        Detecta si la pantalla de Revive está visible.

        Retorna:
            es_gameover : bool
            confianza   : float (0-1)
            frame_resultado : frame con anotaciones
        """
        if frame is None:
            return False, 0.0, frame

        x0, y0, x1, y1 = ROI
        roi = frame[y0:y1, x0:x1]
        roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        frame_resultado = frame.copy()

        mejor_val = 0
        mejor_loc = None
        mejor_tam = None

        for escala in ESCALAS:
            h_t, w_t = self.template.shape
            nw = int(w_t * escala)
            nh = int(h_t * escala)
            if nw >= roi_gris.shape[1] or nh >= roi_gris.shape[0]:
                continue

            t_scaled = cv2.resize(self.template, (nw, nh))
            res = cv2.matchTemplate(roi_gris, t_scaled, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(res)

            if val > mejor_val:
                mejor_val = val
                mejor_loc = loc
                mejor_tam = (nw, nh)

        es_gameover = mejor_val >= UMBRAL

        if es_gameover and mejor_loc is not None:
            mx, my = mejor_loc
            w_m, h_m = mejor_tam
            x_real = x0 + mx
            y_real = y0 + my
            cv2.rectangle(frame_resultado,
                          (x_real, y_real), (x_real + w_m, y_real + h_m),
                          (0, 0, 255), 2)
            cv2.putText(frame_resultado,
                        f"GAME OVER detectado (conf={mejor_val:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame_resultado,
                        f"Jugando (max={mejor_val:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        return es_gameover, mejor_val, frame_resultado

    # ─────────────────────────────────────────
    def probar(self):
        print("=== DETECTOR DE GAME OVER ===")
        print("Muere en el juego para probar la detección")
        print("Presiona 'q' para salir")
        time.sleep(2)

        cv2.namedWindow("GameOver Detector")

        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                print("Esperando BlueStacks...")
                time.sleep(1)
                continue

            es_gameover, confianza, frame_resultado = self.detectar_gameover(frame)

            if es_gameover:
                print(f"💀 GAME OVER detectado — conf={confianza:.2f}")

            cv2.imshow("GameOver Detector", frame_resultado)
            cv2.moveWindow("GameOver Detector", 100, 100)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorGameOver()
    detector.probar()