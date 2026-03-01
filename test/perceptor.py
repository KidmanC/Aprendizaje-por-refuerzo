"""
perceptor.py — Módulo de percepción integrado para Banana Kong RL
Resolución BlueStacks: 960x540

Captura el frame UNA SOLA VEZ por ciclo y lo pasa a todos los detectores.
Retorna un diccionario de estado listo para el agente de RL.

Uso:
    perceptor = Perceptor()
    estado = perceptor.get_estado()

Estado retornado:
    {
        "kong"      : (cx, cy) normalizados o None,
        "kong_pose" : str o None,
        "bananas"   : [(cx, cy), ...],
        "barriles"  : [(cx, cy), ...],
        "rocas"     : [(cx, cy, tipo), ...],
        "agua"      : bool,
        "agua_cx"   : float o None,
        "muros"     : [{"tipo", "cx", "cy", "altura", "ancho"}, ...],
        "game_over" : bool,
        "frame"     : np.ndarray (frame crudo, útil para debug),
    }
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time

# Importar todos los detectores
from detector_kong      import DetectorKong
from detector_bananas   import DetectorBananas
from detector_barriles  import DetectorBarriles
from detector_rocas     import DetectorRocas
from detector_agua      import DetectorAgua
from detector_muros     import DetectorMuros
from detector_gameover  import DetectorGameOver


# ── Optimización: no correr detectores costosos en cada frame ────────
# Game over solo se verifica cada N frames (es lento y raro)
GAMEOVER_CADA_N_FRAMES = 10


class Perceptor:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()

        print("Cargando detectores...")
        self.det_kong     = DetectorKong()
        self.det_bananas  = DetectorBananas()
        self.det_barriles = DetectorBarriles()
        self.det_rocas    = DetectorRocas()
        self.det_agua     = DetectorAgua()
        self.det_muros    = DetectorMuros()
        self.det_gameover = DetectorGameOver()
        print("✅ Todos los detectores listos")

        self._frame_count = 0
        self._ultimo_gameover = False

    # ─────────────────────────────────────────
    def actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle(self.titulo)
        if ventanas:
            self.ventana = ventanas[0]
            self.monitor = {
                "top":    self.ventana.top,
                "left":   self.ventana.left,
                "width":  self.ventana.width,
                "height": self.ventana.height,
            }
            return True
        return False

    def _capturar_frame(self):
        """Captura el frame una sola vez por ciclo."""
        if self.ventana is None:
            if not self.actualizar_ventana():
                return None
        try:
            self.monitor = {
                "top":    self.ventana.top,
                "left":   self.ventana.left,
                "width":  self.ventana.width,
                "height": self.ventana.height,
            }
        except Exception:
            self.ventana = None
            return None
        screenshot = self.sct.grab(self.monitor)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

    # ─────────────────────────────────────────
    def get_estado(self):
        """
        Captura el frame y corre todos los detectores.
        Retorna el estado completo del juego como diccionario.
        """
        self._frame_count += 1
        frame = self._capturar_frame()

        if frame is None:
            return self._estado_vacio()

        # ── Kong ─────────────────────────────────────────────────────
        kong_pos, _, kong_pose = self.det_kong.detectar_kong(frame)

        # ── Bananas ──────────────────────────────────────────────────
        cantidad, _, contornos_validos = self.det_bananas.detectar_bananas(frame)
        h_f, w_f = frame.shape[:2]
        bananas_pos = []
        for cnt in contornos_validos:
            x, y, bw, bh = cv2.boundingRect(cnt)
            bananas_pos.append(((x + bw/2) / w_f, (y + bh/2) / h_f))
        bananas = {"cantidad": cantidad, "posiciones": bananas_pos}

        # ── Barriles ─────────────────────────────────────────────────
        barriles, _, _ = self.det_barriles.detectar_barriles(frame)

        # ── Rocas ────────────────────────────────────────────────────
        rocas, _ = self.det_rocas.detectar_rocas(frame)

        # ── Agua ─────────────────────────────────────────────────────
        hay_agua, agua_cx, _, _ = self.det_agua.detectar_agua(frame)

        # ── Muros ────────────────────────────────────────────────────
        muros_raw, _, _ = self.det_muros.detectar_muros(frame)
        muros = [
            {
                "tipo":   m["tipo"],
                "cx":     m["cx"],
                "cy":     m["cy"],
                "altura": m["altura"],
                "ancho":  m["ancho"],
            }
            for m in muros_raw
        ]

        # ── Game Over (solo cada N frames) ───────────────────────────
        if self._frame_count % GAMEOVER_CADA_N_FRAMES == 0:
            game_over, _, _ = self.det_gameover.detectar_gameover(frame)
            self._ultimo_gameover = game_over
        else:
            game_over = self._ultimo_gameover

        return {
            "kong":       kong_pos,
            "kong_pose":  kong_pose,
            "bananas":    bananas,    # dict: {cantidad, posiciones}
            "barriles":   barriles,
            "rocas":      rocas,
            "agua":       hay_agua,
            "agua_cx":    agua_cx,
            "muros":      muros,
            "game_over":  game_over,
            "frame":      frame,
        }

    def _estado_vacio(self):
        return {
            "kong":      None,
            "kong_pose": None,
            "bananas":   {"cantidad": 0, "posiciones": []},
            "barriles":  [],
            "rocas":     [],
            "agua":      False,
            "agua_cx":   None,
            "muros":     [],
            "game_over": False,
            "frame":     None,
        }

    # ─────────────────────────────────────────
    def probar(self):
        """Modo debug — muestra el estado en consola y el frame anotado."""
        print("=== PERCEPTOR INTEGRADO ===")
        print("q=salir")
        time.sleep(2)

        cv2.namedWindow("Perceptor")
        fps_t = time.time()
        fps_c = 0

        while True:
            estado = self.get_estado()
            frame = estado["frame"]
            if frame is None:
                print("Esperando BlueStacks...")
                time.sleep(1)
                continue

            # Anotar frame con todo el estado
            frame_debug = frame.copy()
            self._dibujar_estado(frame_debug, estado)

            fps_c += 1
            if time.time() - fps_t >= 1.0:
                fps = fps_c / (time.time() - fps_t)
                print(f"FPS: {fps:.1f} | Kong: {estado['kong']} | "
                      f"Bananas: {estado['bananas']['cantidad']} | "
                      f"Barriles: {len(estado['barriles'])} | "
                      f"Agua: {estado['agua']} | "
                      f"GameOver: {estado['game_over']}")
                fps_c = 0
                fps_t = time.time()

            cv2.imshow("Perceptor", frame_debug)
            cv2.moveWindow("Perceptor", 100, 100)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def _dibujar_estado(self, frame, estado):
        """Dibuja todas las detecciones sobre el frame para debug."""
        h, w = frame.shape[:2]

        # Kong
        if estado["kong"]:
            cx, cy = estado["kong"]
            cv2.circle(frame, (int(cx*w), int(cy*h)), 8, (0, 165, 255), -1)
            cv2.putText(frame, f"Kong [{estado['kong_pose']}]",
                        (int(cx*w)+10, int(cy*h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)

        # Bananas
        for cx, cy in estado["bananas"]["posiciones"]:
            cv2.circle(frame, (int(cx*w), int(cy*h)), 6, (0, 255, 255), 2)
        cv2.putText(frame, f"Bananas: {estado['bananas']['cantidad']}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Barriles
        for cx, cy in estado["barriles"]:
            cv2.circle(frame, (int(cx*w), int(cy*h)), 8, (0, 0, 255), 2)

        # Rocas
        for cx, cy, tipo in estado["rocas"]:
            cv2.circle(frame, (int(cx*w), int(cy*h)), 8, (0, 140, 255), 2)

        # Agua
        if estado["agua"]:
            cv2.putText(frame, "AGUA", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # Muros
        for m in estado["muros"]:
            cx, cy = m["cx"], m["cy"]
            cv2.circle(frame, (int(cx*w), int(cy*h)), 8, (200, 200, 200), 2)

        # Game Over
        if estado["game_over"]:
            cv2.putText(frame, "GAME OVER", (w//2-80, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


if __name__ == "__main__":
    perceptor = Perceptor()
    perceptor.probar()