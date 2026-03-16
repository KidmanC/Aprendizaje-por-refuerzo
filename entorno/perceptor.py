"""
perceptor.py — Perceptor con hilo de detección en background.

El agente NUNCA espera a los detectores — lee el último estado disponible.
Los detectores corren continuamente en un hilo separado.
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import threading

from deteccion.detector_kong     import DetectorKong
from deteccion.detector_bananas  import DetectorBananas
from deteccion.detector_gameover import DetectorGameOver

GAMEOVER_CADA = 10


class Perceptor:
    def __init__(self):
        self.titulo  = "BlueStacks"
        self.ventana = None
        self.actualizar_ventana()

        print("Cargando detectores...")
        self.det_kong     = DetectorKong()
        self.det_bananas  = DetectorBananas()
        self.det_gameover = DetectorGameOver()
        print("✅ Detectores listos (modo simplificado: kong + bananas)")

        # Estado compartido entre hilo y agente
        self._estado      = self._estado_vacio()
        self._lock        = threading.Lock()
        self._frame_count = 0
        self._ultimo_gameover = False

        # Arrancar hilo de detección
        self._activo = True
        self._hilo   = threading.Thread(target=self._loop, daemon=True)
        self._hilo.start()

        # Esperar primer estado válido antes de retornar
        print("Esperando primer frame...", end=" ")
        for _ in range(50):
            time.sleep(0.1)
            with self._lock:
                if self._estado["frame"] is not None:
                    break
        print("listo")

        # Colisiones — detectadas en el hilo, leídas por el entorno
        self._bananas_recogidas   = 0   # acumulador entre calls
        self._pico_colisiones     = 0
        self.MARGEN_KONG          = 10

        # Hilo de display — muestra el estado en tiempo real
        self._display_activo  = False
        self._total_bananas   = 0
        self._step_count      = 0

    # ── Ventana ──────────────────────────────────────────────────────
    def actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle("BlueStacks")
        if ventanas:
            self.ventana = ventanas[0]
            return True
        return False

    def _get_monitor(self):
        return {
            "top":    self.ventana.top,
            "left":   self.ventana.left,
            "width":  self.ventana.width,
            "height": self.ventana.height,
        }

    # ── Hilo de detección ─────────────────────────────────────────────
    def _loop(self):
        """Corre continuamente en background. mss propio por thread-safety."""
        sct = mss()
        while self._activo:
            try:
                if self.ventana is None:
                    self.actualizar_ventana()
                    time.sleep(0.5)
                    continue

                screenshot = sct.grab(self._get_monitor())
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

                # Detectar
                kong_pos, _, kong_pose, kong_rect, _ = self.det_kong.detectar_kong(frame)
                cantidad, _, contornos, rects = self.det_bananas.detectar_bananas(frame)

                h_f, w_f = frame.shape[:2]
                bananas_pos = []
                for cnt in contornos:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    bananas_pos.append(((x + bw/2) / w_f, (y + bh/2) / h_f))

                self._frame_count += 1
                if self._frame_count % GAMEOVER_CADA == 0:
                    game_over, _, _ = self.det_gameover.detectar_gameover(frame)
                    self._ultimo_gameover = game_over

                estado = {
                    "kong":      kong_pos,
                    "kong_rect": kong_rect,
                    "kong_pose": kong_pose,
                    "bananas":   {"cantidad": cantidad, "posiciones": bananas_pos, "rects": rects},
                    "barriles":  [],
                    "muros":     [],
                    "agua":      False,
                    "game_over": self._ultimo_gameover,
                    "frame":     frame,
                }

                # Detectar colisiones en el hilo — corre a máxima velocidad
                self._detectar_colisiones(kong_rect, rects)

                with self._lock:
                    self._estado = estado

            except Exception as e:
                print(f"[perceptor] error en hilo: {e}")
                time.sleep(0.1)

    # ── Colisiones ───────────────────────────────────────────────────
    def _detectar_colisiones(self, kong_rect, rects_bananas):
        """Corre en el hilo — detecta patrón pico→cero continuamente."""
        if kong_rect is None:
            return
        kx, ky, kw, kh = kong_rect
        kx -= self.MARGEN_KONG;  ky -= self.MARGEN_KONG
        kw += self.MARGEN_KONG * 2; kh += self.MARGEN_KONG * 2

        colisiones_ahora = 0
        for (bx, by, bw, bh) in rects_bananas:
            if kx < bx+bw and kx+kw > bx and ky < by+bh and ky+kh > by:
                colisiones_ahora += 1

        if colisiones_ahora > self._pico_colisiones:
            self._pico_colisiones = colisiones_ahora
        if colisiones_ahora == 0 and self._pico_colisiones > 0:
            self._bananas_recogidas += self._pico_colisiones
            self._pico_colisiones    = 0

    def pop_bananas_recogidas(self):
        """El entorno llama esto en cada step para leer y resetear el contador."""
        with self._lock:
            n = self._bananas_recogidas
            self._bananas_recogidas = 0
            return n

    def reset_colisiones(self):
        """Llamar al inicio de cada episodio."""
        self._bananas_recogidas = 0
        self._pico_colisiones   = 0

    # ── API pública ───────────────────────────────────────────────────
    def get_estado(self):
        """Retorna el último estado calculado — instantáneo."""
        with self._lock:
            return dict(self._estado)

    def get_conteo_bananas(self):
        with self._lock:
            return self._estado["bananas"]["cantidad"]

    def start_display(self):
        """Arranca ventana de visualización en su propio hilo."""
        self._display_activo = True
        self._hilo_display = threading.Thread(target=self._loop_display, daemon=True)
        self._hilo_display.start()

    def _loop_display(self):
        """
        Muestra el estado en tiempo real a la mitad de resolución.
        Escalar antes de dibujar = 4x menos píxeles procesados.
        """
        cv2.namedWindow("Debug")
        cv2.moveWindow("Debug", 1000, 100)
        ESCALA = 0.5

        while self._display_activo:
            estado = self.get_estado()
            frame  = estado["frame"]
            if frame is None:
                time.sleep(0.05)
                continue

            # Escalar primero — dibujar sobre imagen pequeña
            small = cv2.resize(frame, (0, 0), fx=ESCALA, fy=ESCALA)

            # Bounding box Kong
            if estado["kong_rect"]:
                kx, ky, kw, kh = [int(v * ESCALA) for v in estado["kong_rect"]]
                cv2.rectangle(small, (kx, ky), (kx+kw, ky+kh), (0, 165, 255), 1)

            # Bounding boxes bananas
            for bx, by, bw, bh in estado["bananas"]["rects"]:
                bx, by, bw, bh = [int(v * ESCALA) for v in (bx, by, bw, bh)]
                cv2.rectangle(small, (bx, by), (bx+bw, by+bh), (0, 255, 0), 1)

            # Panel info
            cv2.rectangle(small, (3, 3), (200, 50), (0, 0, 0), -1)
            cv2.putText(small, f"Bananas: {self._total_bananas}  Step: {self._step_count}",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if estado["game_over"]:
                cv2.putText(small, "GAME OVER", (6, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Debug", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyWindow("Debug")

    def parar(self):
        self._activo = False
        self._display_activo = False

    # ── Estado vacío ──────────────────────────────────────────────────
    def _estado_vacio(self):
        return {
            "kong":      None,
            "kong_rect": None,
            "kong_pose": None,
            "bananas":   {"cantidad": 0, "posiciones": [], "rects": []},
            "barriles":  [],
            "muros":     [],
            "agua":      False,
            "game_over": False,
            "frame":     None,
        }

    # ── Debug ─────────────────────────────────────────────────────────
    def probar(self):
        print("=== PERCEPTOR ===  q=salir")
        time.sleep(2)
        cv2.namedWindow("Perceptor")
        fps_t = time.time()
        fps_c = 0

        while True:
            estado = self.get_estado()
            frame  = estado["frame"]
            if frame is None:
                time.sleep(0.05)
                continue

            debug = frame.copy()
            if estado["kong_rect"]:
                kx, ky, kw, kh = estado["kong_rect"]
                cv2.rectangle(debug, (kx, ky), (kx+kw, ky+kh), (0,165,255), 2)
            for bx, by, bw, bh in estado["bananas"]["rects"]:
                cv2.rectangle(debug, (bx, by), (bx+bw, by+bh), (0,255,0), 2)
            cv2.rectangle(debug, (5,5), (280,45), (0,0,0), -1)
            cv2.putText(debug, f"Bananas: {estado['bananas']['cantidad']}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            fps_c += 1
            if time.time() - fps_t >= 1.0:
                fps = fps_c / (time.time() - fps_t)
                print(f"FPS display: {fps:.1f} | Kong: {estado['kong']} | "
                      f"Bananas: {estado['bananas']['cantidad']} | GO: {estado['game_over']}")
                fps_c = 0
                fps_t = time.time()

            cv2.imshow("Perceptor", debug)
            cv2.moveWindow("Perceptor", 100, 100)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        self.parar()


if __name__ == "__main__":
    p = Perceptor()
    p.probar()