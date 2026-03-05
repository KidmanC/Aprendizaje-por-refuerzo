"""
detector_kong.py — Detección de Kong: HSV + Template Matching
Resolución BlueStacks: 960x540

Estrategia:
  1. HSV encuentra blobs marrones del tamaño correcto dentro de la ROI
  2. Template matching corre SOLO sobre cada blob candidato
  3. El blob con mejor confianza es Kong

Ventaja: el template matching nunca compite contra el fondo verde/amarillo.
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import os


# ── Configuración ────────────────────────────────────────────────────

# ROI donde aparece Kong (izquierda de la pantalla, sin HUD)
ROI = (0, 60, 420, 510)

# Rango HSV para el marrón de Kong
KONG_HSV_BAJO = np.array([5, 60, 50])
KONG_HSV_ALTO = np.array([25, 255, 220])

# Rango de área válida para blobs de Kong (en px² dentro de la ROI)
BLOB_AREA_MIN = 400
BLOB_AREA_MAX = 3500  # reducido para excluir objetos grandes del fondo

# Ratio ancho/alto válido para Kong (ni muy delgado ni muy ancho)
BLOB_RATIO_MIN = 0.4
BLOB_RATIO_MAX = 2.5

# Umbral de confianza para template matching (sobre el blob recortado)
UMBRAL = 0.35

# Escalas a probar en template matching
ESCALAS = [0.9, 1.0, 1.1]

# Margen extra alrededor del blob para el recorte (px)
MARGEN_BLOB = 10

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

TEMPLATES_INFO = [
    ("kong_inicio-bg.png",     "inicio"),
    ("kong_corriendo1-bg.png", "corriendo"),
    ("kong_corriendo3-bg.png", "corriendo"),
    ("kong_saltando-bg.png",   "saltando"),
    ("kong_saltando2-bg.png",  "saltando"),
    ("kong_paracaidas-bg.png", "paracaidas"),
    ("kong_dash-bg.png",       "dash"),
    ("kong_liana-bg.png",      "liana"),
    ("kong_guacamaya-bg.png",  "guacamaya"),
]


class DetectorKong:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()
        self.posicion_anterior = None

        # Cargar templates con alpha
        self.templates = []
        for filename, pose in TEMPLATES_INFO:
            ruta = os.path.join(TEMPLATES_DIR, filename)
            img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"⚠️  No se encontró: {ruta}")
                continue

            if img.shape[2] == 4:
                alpha = img[:, :, 3]
                img_gris = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                alpha = None

            self.templates.append((img_gris, alpha, pose))

        print(f"✅ {len(self.templates)} templates cargados")

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
        if not self.actualizar_ventana():
            return None
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ─────────────────────────────────────────
    def _encontrar_blobs_hsv(self, roi):
        """
        Usa HSV para encontrar blobs marrones del tamaño correcto.
        Retorna lista de (x, y, w, h) en coordenadas de la ROI.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, KONG_HSV_BAJO, KONG_HSV_ALTO)
        mascara = cv2.erode(mascara, None, iterations=1)
        mascara = cv2.dilate(mascara, None, iterations=2)

        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if not (BLOB_AREA_MIN < area < BLOB_AREA_MAX):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h if h > 0 else 0
            if not (BLOB_RATIO_MIN < ratio < BLOB_RATIO_MAX):
                continue
            blobs.append((x, y, w, h))

        return blobs, mascara

    # ─────────────────────────────────────────
    def _match_sobre_blob(self, roi_gris, x, y, w, h):
        """
        Corre template matching sobre el recorte del blob.
        Retorna (mejor_val, mejor_pose).
        """
        # Recortar blob con margen
        h_roi, w_roi = roi_gris.shape
        bx0 = max(0, x - MARGEN_BLOB)
        by0 = max(0, y - MARGEN_BLOB)
        bx1 = min(w_roi, x + w + MARGEN_BLOB)
        by1 = min(h_roi, y + h + MARGEN_BLOB)
        recorte = roi_gris[by0:by1, bx0:bx1]

        mejor_val = 0
        mejor_pose = None

        for template, alpha, pose in self.templates:
            h_t, w_t = template.shape
            for escala in ESCALAS:
                nw = int(w_t * escala)
                nh = int(h_t * escala)
                if nw >= recorte.shape[1] or nh >= recorte.shape[0]:
                    continue

                t_scaled = cv2.resize(template, (nw, nh))

                if alpha is not None:
                    a_scaled = cv2.resize(alpha, (nw, nh))
                    res = cv2.matchTemplate(
                        recorte, t_scaled, cv2.TM_SQDIFF_NORMED, mask=a_scaled
                    )
                else:
                    res = cv2.matchTemplate(recorte, t_scaled, cv2.TM_SQDIFF_NORMED)

                min_val, _, _, _ = cv2.minMaxLoc(res)
                val = 1.0 - min_val

                if val > mejor_val:
                    mejor_val = val
                    mejor_pose = pose

        return mejor_val, mejor_pose

    # ─────────────────────────────────────────
    def detectar_kong(self, frame):
        """
        Detecta a Kong combinando HSV (candidatos) + Template Matching (verificación).

        Retorna:
            posicion      : (cx, cy) normalizados a [0,1] o None
            frame_resultado : frame con anotaciones
            pose          : string con la pose detectada o None
        """
        if frame is None:
            return None, frame, None

        x0, y0, x1, y1 = ROI
        roi = frame[y0:y1, x0:x1]
        roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = roi.shape[:2]

        frame_resultado = frame.copy()
        cv2.rectangle(frame_resultado, (x0, y0), (x1, y1), (255, 255, 0), 1)

        # ── Paso 1: HSV encuentra blobs candidatos ───────────────────
        blobs, mascara = self._encontrar_blobs_hsv(roi)

        if not blobs:
            cv2.putText(frame_resultado, "Kong: sin blobs HSV",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return None, frame_resultado, None

        # Dibujar blobs candidatos en gris
        for (bx, by, bw, bh) in blobs:
            cv2.rectangle(frame_resultado,
                          (x0 + bx, y0 + by),
                          (x0 + bx + bw, y0 + by + bh),
                          (180, 180, 180), 1)

        # ── Paso 2: Template matching sobre cada blob ────────────────
        mejor_val_global = 0
        mejor_blob = None
        mejor_pose = None

        for blob in blobs:
            bx, by, bw, bh = blob
            val, pose = self._match_sobre_blob(roi_gris, bx, by, bw, bh)
            if val > mejor_val_global:
                mejor_val_global = val
                mejor_blob = blob
                mejor_pose = pose

        # ── Sin match suficientemente bueno ─────────────────────────
        if mejor_val_global < UMBRAL or mejor_blob is None:
            cv2.putText(frame_resultado,
                        f"Kong: NO detectado (max={mejor_val_global:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return None, frame_resultado, None

        # ── Calcular posición final ───────────────────────────────────
        bx, by, bw, bh = mejor_blob
        x_real = x0 + bx
        y_real = y0 + by

        cx = (x_real + bw / 2) / frame.shape[1]
        cy = (y_real + bh / 2) / frame.shape[0]
        self.posicion_anterior = (cx, cy)

        # Dibujar detección final
        cv2.rectangle(frame_resultado,
                      (x_real, y_real), (x_real + bw, y_real + bh),
                      (0, 165, 255), 2)
        cv2.circle(frame_resultado,
                   (int(x_real + bw / 2), int(y_real + bh / 2)),
                   5, (0, 165, 255), -1)
        cv2.putText(frame_resultado,
                    f"Kong [{mejor_pose}] cx={cx:.2f} cy={cy:.2f} conf={mejor_val_global:.2f}",
                    (x_real, y_real - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        return (cx, cy), frame_resultado, mejor_pose

    # ─────────────────────────────────────────
    def probar(self):
        print("=== DETECTOR DE KONG (HSV + Template Matching) ===")
        print("Presiona 'q' para salir")
        print("Presiona 's' para guardar frame")
        print("Presiona 'm' para mostrar/ocultar máscara HSV")
        time.sleep(2)

        mostrar_mascara = False
        cv2.namedWindow("Kong Detector")
        fps_tiempo = time.time()
        fps_contador = 0

        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                print("Esperando BlueStacks...")
                time.sleep(1)
                continue

            posicion, frame_resultado, pose = self.detectar_kong(frame)

            fps_contador += 1
            if time.time() - fps_tiempo >= 1.0:
                fps = fps_contador / (time.time() - fps_tiempo)
                print(f"FPS: {fps:.1f} | Kong: {posicion} | Pose: {pose}")
                fps_contador = 0
                fps_tiempo = time.time()

            cv2.imshow("Kong Detector", frame_resultado)
            cv2.moveWindow("Kong Detector", 100, 100)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("kong_frame.png", frame_resultado)
                print("Frame guardado")
            elif key == ord("m"):
                mostrar_mascara = not mostrar_mascara
                if not mostrar_mascara:
                    cv2.destroyWindow("Mascara HSV")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorKong()
    detector.probar()