# Aprendizaje por Refuerzo para Bot Autónomo en Videojuegos

## Explicación del tema

Este submódulo del proyecto se enfoca en el desarrollo del **módulo de decisión** del bot autónomo utilizando **aprendizaje por refuerzo (Reinforcement Learning, RL)**. A diferencia del enfoque basado en reglas (Grupo 1), este grupo implementará un agente que aprenda a jugar un videojuego a partir de la información visual extraída por el módulo de percepción, utilizando únicamente la pantalla como entrada y simulando acciones de teclado, mouse o control.

El objetivo es que el agente sea capaz de tomar decisiones en tiempo real, maximizando una función de recompensa definida según el juego seleccionado (por ejemplo: puntaje, tiempo de supervivencia, nivel superado, etc.). Se explorarán algoritmos de RL como DQN (Deep Q-Network), PPO (Proximal Policy Optimization) o A3C (Asynchronous Advantage Actor-Critic), adaptados a entornos visuales.

Este enfoque representa un desafío técnico importante, ya que implica integrar un pipeline de percepción visual con un modelo de decisión entrenable, todo bajo restricciones de tiempo real y sin acceso al motor interno del juego.

---

## Enlaces relevantes

- [OpenAI Gym](https://www.gymlibrary.dev/) – Entornos estándar para probar algoritmos de RL.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) – Implementaciones de algoritmos RL en PyTorch.
- [Unity ML-Agents](https://unity.com/products/machine-learning-agents) – Toolkit para entrenar agentes en entornos Unity (útil si se usa un juego accesible).
- [DQN Paper (Mnih et al.)](https://www.nature.com/articles/nature14236) – Artículo original de Deep Q-Networks.
- [PPO Paper (Schulman et al.)](https://arxiv.org/abs/1707.06347) – Algoritmo estable y popular para RL.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) – Implementaciones educativas y modulares de RL.
- [Screen Capture en Python](https://github.com/learncodebygaming/opencv_tutorials) – Tutoriales de captura de pantalla con OpenCV.
- [PyDirectInput](https://github.com/learncodebygaming/pydirectinput) – Librería para simular entradas de teclado y mouse.

---

## Códigos de prueba

### 1. Captura de pantalla básica
```python
import cv2
import numpy as np
from mss import mss

# Captura de pantalla en tiempo real
sct = mss()
monitor = {"top": 0, "left": 0, "width": 800, "height": 600}

while True:
    img = np.array(sct.grab(monitor))
    cv2.imshow("Screen Capture", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

### 2. Simulación de teclas
```python
import pydirectinput
import time

# Ejemplo: presionar tecla W por 1 segundo
pydirectinput.keyDown('w')
time.sleep(1)
pydirectinput.keyUp('w')
```

### 3. Estructura mínima de un agente RL con Stable-Baselines3
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Crear entorno (debe implementar la interfaz gym.Env)
env = make_vec_env("MiEntornoVideojuego-v0", n_envs=1)

# Definir y entrenar agente
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("bot_rl_videojuego")

# Cargar y evaluar
model = PPO.load("bot_rl_videojuego")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
```

### 4. Clase base para entorno personalizado
```python
import gym
from gym import spaces
import numpy as np

class VideojuegoEnv(gym.Env):
    def __init__(self):
        super(VideojuegoEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Ejemplo: 4 acciones posibles
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def step(self, action):
        # Ejecutar acción en el juego
        # Obtener nueva pantalla, recompensa, done, info
        return observation, reward, done, info

    def reset(self):
        # Reiniciar el juego y devolver observación inicial
        return observation

    def render(self, mode="human"):
        pass
```