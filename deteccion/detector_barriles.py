"""
detector_barriles.py — Detección de barriles en Banana Kong
Resolución BlueStacks: 960x540

Estrategia híbrida:
  1. HSV con saturación BAJA para capturar barril y excluir Kong
  2. Template matching sobre blobs candidatos
  3. ROI excluye el tercio izquierdo donde está Kong
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import os

# ROI — empieza donde termina Kong (tercio izquierdo excluido)
ROI = (250, 80, 900, 480)

# HSV del barril — interior brillante con V alto (distingue del suelo oscuro)
BARRIL_HSV_BAJO = np.array([8,  80, 160])
BARRIL_HSV_ALTO = np.array([22, 240, 255])

BARRIL_AREA_MIN = 600
BARRIL_AREA_MAX = 6000
RATIO_MIN = 0.6
RATIO_MAX = 1.6
SOLIDEZ_MIN = 0.60
MARGEN_BLOB = 12
UMBRAL = 0.75
ESCALAS = [0.8, 0.9, 1.0, 1.1, 1.2]
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


class DetectorBarriles:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()

        ruta = os.path.join(TEMPLATES_DIR, "barril-bg.png")
        img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"No se encontró: {ruta}")

        if img.shape[2] == 4:
            self.alpha = img[:, :, 3]
            self.template = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            self.template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.alpha = None

        print(f"✅ Template barril: {self.template.shape[1]}x{self.template.shape[0]}px")

    def actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle(self.titulo)
        if ventanas:
            self.ventana = ventanas[0]
            self.monitor = {
                "top": self.ventana.top, "left": self.ventana.left,
                "width": self.ventana.width, "height": self.ventana.height,
            }
            return True
        return False

    def capturar_pantalla(self):
        if self.ventana is None:
            if not self.actualizar_ventana():
                return None
        try:
            self.monitor = {
                "top": self.ventana.top, "left": self.ventana.left,
                "width": self.ventana.width, "height": self.ventana.height,
            }
        except Exception:
            self.ventana = None
            return None
        screenshot = self.sct.grab(self.monitor)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

    def _encontrar_blobs_hsv(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, BARRIL_HSV_BAJO, BARRIL_HSV_ALTO)
        mascara = cv2.erode(mascara, None, iterations=1)
        mascara = cv2.dilate(mascara, None, iterations=2)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if not (BARRIL_AREA_MIN < area < BARRIL_AREA_MAX):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h if h > 0 else 0
            if not (RATIO_MIN < ratio < RATIO_MAX):
                continue
            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)
            solidez = area / area_hull if area_hull > 0 else 0
            if solidez < SOLIDEZ_MIN:
                continue
            blobs.append((x, y, w, h))
        return blobs, mascara

    def _match_sobre_blob(self, roi_gris, x, y, w, h):
        h_roi, w_roi = roi_gris.shape
        bx0 = max(0, x - MARGEN_BLOB)
        by0 = max(0, y - MARGEN_BLOB)
        bx1 = min(w_roi, x + w + MARGEN_BLOB)
        by1 = min(h_roi, y + h + MARGEN_BLOB)
        recorte = roi_gris[by0:by1, bx0:bx1]

        mejor_val = 0
        for escala in ESCALAS:
            h_t, w_t = self.template.shape
            nw, nh = int(w_t * escala), int(h_t * escala)
            if nw >= recorte.shape[1] or nh >= recorte.shape[0]:
                continue
            t_scaled = cv2.resize(self.template, (nw, nh))
            if self.alpha is not None:
                a_scaled = cv2.resize(self.alpha, (nw, nh))
                res = cv2.matchTemplate(recorte, t_scaled, cv2.TM_SQDIFF_NORMED, mask=a_scaled)
            else:
                res = cv2.matchTemplate(recorte, t_scaled, cv2.TM_SQDIFF_NORMED)
            min_val, _, _, _ = cv2.minMaxLoc(res)
            val = 1.0 - min_val
            if val > mejor_val:
                mejor_val = val
        return mejor_val

    def detectar_barriles(self, frame):
        if frame is None:
            return [], frame, None

        x0, y0, x1, y1 = ROI
        roi = frame[y0:y1, x0:x1]
        roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        frame_resultado = frame.copy()
        cv2.rectangle(frame_resultado, (x0, y0), (x1, y1), (255, 255, 0), 1)

        blobs, mascara = self._encontrar_blobs_hsv(roi)

        barriles = []
        for (bx, by, bw, bh) in blobs:
            cv2.rectangle(frame_resultado,
                          (x0+bx, y0+by), (x0+bx+bw, y0+by+bh),
                          (180, 180, 180), 1)

            val = self._match_sobre_blob(roi_gris, bx, by, bw, bh)
            if val < UMBRAL:
                continue

            x_real, y_real = bx + x0, by + y0
            cx = (x_real + bw/2) / frame.shape[1]
            cy = (y_real + bh/2) / frame.shape[0]
            barriles.append((cx, cy))

            cv2.rectangle(frame_resultado, (x_real, y_real), (x_real+bw, y_real+bh), (0, 0, 255), 2)
            cv2.putText(frame_resultado, f"Barril {val:.2f}", (x_real, y_real-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.putText(frame_resultado, f"Barriles: {len(barriles)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,0,255) if barriles else (0,200,0), 2)

        return barriles, frame_resultado, mascara

    def probar(self):
        print("=== DETECTOR DE BARRILES (Híbrido) ===")
        print("q=salir | s=guardar | m=máscara HSV")
        time.sleep(2)

        mostrar_mascara = True
        cv2.namedWindow("Barriles Detector")

        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                time.sleep(1)
                continue

            barriles, frame_resultado, mascara = self.detectar_barriles(frame)
            if barriles:
                print(f"🛢️  {len(barriles)} barril(es)")

            cv2.imshow("Barriles Detector", frame_resultado)
            cv2.moveWindow("Barriles Detector", 100, 100)

            if mostrar_mascara and mascara is not None:
                cv2.imshow("Mascara HSV", mascara)
                cv2.moveWindow("Mascara HSV", 800, 100)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("barril_frame.png", frame_resultado)
                if mascara is not None:
                    cv2.imwrite("barril_mascara.png", mascara)
                print("Guardado")
            elif key == ord("m"):
                mostrar_mascara = not mostrar_mascara
                if not mostrar_mascara:
                    cv2.destroyWindow("Mascara HSV")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorBarriles()
    detector.probar()