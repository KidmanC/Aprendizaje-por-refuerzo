"""
entorno.py — Entorno Gymnasium para Banana Kong RL
Resolución BlueStacks: 960x540

Observación (vector de 24 valores normalizados [0,1]):
    [0-1]   kong_cx, kong_cy
    [2-7]   barril1_cx, barril1_cy, barril2_cx, barril2_cy, barril3_cx, barril3_cy
    [8-13]  banana1_cx, banana1_cy, banana2_cx, banana2_cy, banana3_cx, banana3_cy
    [14-17] muro1_cx, muro1_cy, muro2_cx, muro2_cy
    [18]    hay_agua (0 o 1)
    [19-23] kong_pose one-hot (inicio, corriendo, saltando, paracaidas, dash)

Acciones (Discrete 4):
    0 - NADA
    1 - PLANEAR (press corto=saltar, mantener=planear)
    2 - DASH
    3 - BAJAR

Recompensas:
    +1.0  por banana recogida
    +0.01 por sobrevivir cada step
    -10.0 por game over
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pyautogui

from entorno.perceptor import Perceptor
from controles.acciones  import ModuloAcciones, NADA, PLANEAR, DASH, BAJAR


# ── Configuración ────────────────────────────────────────────────────
MAX_STEPS       = 2000   # máximo de steps por episodio
DELAY_ACCION    = 0.05   # segundos entre steps (controla velocidad del agente)
DELAY_REINICIO  = 3.0    # segundos para esperar que el juego reinicie

OBS_SIZE = 24

# Poses de Kong → índice one-hot
POSES = {"inicio": 0, "corriendo": 1, "saltando": 2, "paracaidas": 3, "dash": 4}


class BananaKongEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Espacio de observación — vector normalizado
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        # Espacio de acciones
        self.action_space = spaces.Discrete(4)

        # Módulos
        self.perceptor = Perceptor()
        self.acciones  = ModuloAcciones()

        self._step_count   = 0
        self._bananas_prev = 0
        self._primer_episodio = True  # no reiniciar en el primer reset

    # ─────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pyautogui.mouseUp()

        if self._primer_episodio:
            # Primer reset: el juego ya está corriendo, no hacer nada
            self._primer_episodio = False
        else:
            # Game over real: navegar pantallas de reinicio
            self._reiniciar_juego()

        self._step_count   = 0
        self._bananas_prev = 0

        estado = self.perceptor.get_estado()
        obs    = self._estado_a_obs(estado)
        return obs, {}

    # ─────────────────────────────────────────
    def step(self, accion):
        # Ejecutar acción
        self.acciones.ejecutar(accion)
        time.sleep(DELAY_ACCION)

        # Observar nuevo estado
        estado = self.perceptor.get_estado()
        obs    = self._estado_a_obs(estado)

        # ── Recompensa ───────────────────────────────────────────────
        reward = 0.0

        # +0.01 por sobrevivir
        reward += 0.01

        # +1.0 por cada banana nueva recogida
        bananas_ahora = estado["bananas"]["cantidad"]
        bananas_antes = self._bananas_prev
        if bananas_ahora < self._bananas_prev:
            # Menos bananas visibles = Kong pasó por encima de ellas
            reward += 1.0 * (self._bananas_prev - bananas_ahora)
        self._bananas_prev = bananas_ahora

        
        # -10.0 por game over (ignorar los primeros 60 steps para evitar falsos positivos)
        terminated = False
        if estado["game_over"] and self._step_count > 60:
            reward     -= 10.0
            terminated  = True

        # Log

        if bananas_ahora < bananas_antes:
            print(f'  [step {self._step_count}] BANANA RECOGIDA | antes={bananas_antes} ahora={bananas_ahora} | reward={reward:+.2f}')
        if terminated:
            print(f'  [step {self._step_count}] GAME OVER | reward={reward:+.2f}')
            
        # Truncar si se supera el máximo de steps
        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS

        info = {
            "kong":     estado["kong"],
            "bananas":  bananas_ahora,
            "barriles": len(estado["barriles"]),
            "agua":     estado["agua"],
        }

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────
    def _estado_a_obs(self, estado):
        """Convierte el estado del perceptor a vector de observación."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # Kong posición [0-1]
        if estado["kong"]:
            obs[0], obs[1] = estado["kong"]
        else:
            obs[0], obs[1] = 0.5, 0.5  # centro como fallback

        # Barriles más cercanos a Kong [2-7]
        barriles = self._ordenar_por_distancia(estado["barriles"], obs[0])
        for i, (cx, cy) in enumerate(barriles[:3]):
            obs[2 + i*2] = cx
            obs[3 + i*2] = cy

        # Bananas más cercanas [8-13]
        bananas = self._ordenar_por_distancia(
            estado["bananas"]["posiciones"], obs[0]
        )
        for i, (cx, cy) in enumerate(bananas[:3]):
            obs[8  + i*2] = cx
            obs[9  + i*2] = cy

        # Muros más cercanos [14-17]
        muros = self._ordenar_por_distancia(
            [(m["cx"], m["cy"]) for m in estado["muros"]], obs[0]
        )
        for i, (cx, cy) in enumerate(muros[:2]):
            obs[14 + i*2] = cx
            obs[15 + i*2] = cy

        # Agua [18]
        obs[18] = 1.0 if estado["agua"] else 0.0

        # Pose de Kong one-hot [19-23]
        pose_idx = POSES.get(estado["kong_pose"], 1)  # default: corriendo
        obs[19 + pose_idx] = 1.0

        return obs

    def _ordenar_por_distancia(self, lista, kong_cx):
        """Ordena elementos por distancia horizontal a Kong."""
        if not lista:
            return []
        return sorted(lista, key=lambda p: abs(p[0] - kong_cx))

    # ─────────────────────────────────────────
    def _reiniciar_juego(self):
        """
        Maneja el flujo completo de reinicio:
          1. Espera que pantalla Revive se cierre sola
          2. Clica flecha en Next Reward
          3. Clica Play Again
        """
        import os
        tpl_dir = os.path.join(os.path.dirname(__file__), "templates")
        tpl_flecha     = self._cargar_template(os.path.join(tpl_dir, "flecha.png"))
        tpl_play_again = self._cargar_template(os.path.join(tpl_dir, "play_again.png"))

        print("Esperando cierre de pantalla Revive...")
        time.sleep(4.5)

        print("Buscando flecha Next Reward...")
        if not self._esperar_y_clicar(tpl_flecha, timeout=8.0, etiqueta="flecha"):
            print("Flecha no encontrada, continuando...")

        time.sleep(1.0)

        print("Buscando Play Again...")
        if not self._esperar_y_clicar(tpl_play_again, timeout=8.0, etiqueta="Play Again"):
            print("Play Again no encontrado, continuando...")

        time.sleep(DELAY_REINICIO)
        print("Juego reiniciado")

    def _cargar_template(self, ruta):
        import cv2
        img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            gris  = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gris  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alpha = None
        return (gris, alpha)

    def _esperar_y_clicar(self, template, timeout=8.0, etiqueta=""):
        import cv2
        import numpy as np
        if template is None:
            return False
        t_gris, t_alpha = template
        inicio = time.time()
        while time.time() - inicio < timeout:
            estado = self.perceptor.get_estado()
            frame  = estado["frame"]
            if frame is None:
                time.sleep(0.2)
                continue
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for escala in [0.9, 1.0, 1.1]:
                h_t, w_t = t_gris.shape
                nw, nh = int(w_t * escala), int(h_t * escala)
                if nw >= gris.shape[1] or nh >= gris.shape[0]:
                    continue
                t_s = cv2.resize(t_gris, (nw, nh))
                if t_alpha is not None:
                    a_s = cv2.resize(t_alpha, (nw, nh))
                    res = cv2.matchTemplate(gris, t_s, cv2.TM_CCOEFF_NORMED, mask=a_s)
                else:
                    res = cv2.matchTemplate(gris, t_s, cv2.TM_CCOEFF_NORMED)
                _, val, _, loc = cv2.minMaxLoc(res)
                if val >= 0.65:
                    cx = loc[0] + nw // 2 + self.perceptor.ventana.left
                    cy = loc[1] + nh // 2 + self.perceptor.ventana.top
                    print(f"  {etiqueta} encontrado (conf={val:.2f})")
                    pyautogui.click(cx, cy)
                    return True
            time.sleep(0.3)
        return False

    # ─────────────────────────────────────────
    def render(self):
        pass  # La visualización la maneja el perceptor en modo debug

    def close(self):
        pyautogui.mouseUp()
        print("Entorno cerrado")