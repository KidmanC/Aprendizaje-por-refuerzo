"""
acciones.py — Módulo de acciones para Banana Kong RL
Resolución BlueStacks: 960x540

Teclas configuradas en BlueStacks Game Controls:
    W → saltar / planear
    D → dash
    S → bajar

Acciones:
    0 - NADA    : no hacer nada
    1 - PLANEAR : tap W = saltar, mantener W = planear
    2 - DASH    : tap D
    3 - BAJAR   : tap S
"""

import pyautogui
import pygetwindow as gw
import time
import signal
import sys

pyautogui.FAILSAFE = False

# ── Teclas configuradas en BlueStacks ────────────────────────────────
TECLA_SALTAR  = 'w'
TECLA_DASH    = 'd'
TECLA_BAJAR   = 's'

# Duración del press para planear (mantener W)
DURACION_PLANEAR = 0.4   # segundos

# Acciones
NADA    = 0
PLANEAR = 1
DASH    = 2
BAJAR   = 3

N_ACCIONES = 4


class ModuloAcciones:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self._actualizar_ventana()
        self._planeando = False
        signal.signal(signal.SIGINT, self._parada_emergencia)
        print("✅ Módulo de acciones listo")
        print(f"   W=saltar/planear  D=dash  S=bajar  (tap=saltar, mantener=planear)")
        print("   Ctrl+C para parada de emergencia")

    def _parada_emergencia(self, sig=None, frame=None):
        pyautogui.keyUp(TECLA_SALTAR)
        self._planeando = False
        print("\n🛑 Parada de emergencia")
        sys.exit(0)

    def parar(self):
        self._parada_emergencia()

    def _actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle(self.titulo)
        if ventanas:
            self.ventana = ventanas[0]
            return True
        return False

    def _foco_bluestacks(self):
        if self.ventana is None:
            self._actualizar_ventana()
        try:
            self.ventana.activate()
        except Exception:
            pass

    def ejecutar(self, accion):
        self._foco_bluestacks()

        # Si venía planeando y la nueva acción no es PLANEAR → soltar W
        if self._planeando and accion != PLANEAR:
            pyautogui.keyUp(TECLA_SALTAR)
            self._planeando = False

        if accion == NADA:
            pass

        elif accion == PLANEAR:
            if not self._planeando:
                pyautogui.keyDown(TECLA_SALTAR)
                self._planeando = True
            # Si ya planeaba → mantiene presionado

        elif accion == DASH:
            pyautogui.press(TECLA_DASH)

        elif accion == BAJAR:
            pyautogui.press(TECLA_BAJAR)

    def probar(self):
        print("=== MÓDULO DE ACCIONES ===")
        print("Pon BlueStacks en primer plano y presiona:")
        print("  1 → PLANEAR  |  2 → DASH  |  3 → BAJAR  |  q → salir")
        time.sleep(3)

        import keyboard
        print("Listo")
        try:
            while True:
                if keyboard.is_pressed("1"):
                    self.ejecutar(PLANEAR)
                    time.sleep(0.05)
                elif keyboard.is_pressed("2"):
                    print("→ DASH")
                    self.ejecutar(DASH)
                    time.sleep(0.3)
                elif keyboard.is_pressed("3"):
                    print("→ BAJAR")
                    self.ejecutar(BAJAR)
                    time.sleep(0.3)
                elif keyboard.is_pressed("q"):
                    break
                else:
                    self.ejecutar(NADA)
                    time.sleep(0.05)
        finally:
            pyautogui.keyUp(TECLA_SALTAR)
            print("Módulo cerrado")


if __name__ == "__main__":
    modulo = ModuloAcciones()
    modulo.probar()