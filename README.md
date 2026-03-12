# Bot Autónomo para Banana Kong - Aprendizaje por Refuerzo

**Universidad del Norte - Facultad de Ingeniería de Sistemas**  
**Proyecto Final - Grupo 2 - Aprendizaje por Refuerzo**  
**Kidman Cabana, Santiago Romero - Barranquilla, Colombia - 2026**

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Planteamiento del Problema](#2-planteamiento-del-problema)
3. [Restricciones y Supuestos](#3-restricciones-y-supuestos)
4. [Alcance del Proyecto](#4-alcance-del-proyecto)
5. [Objetivos](#5-objetivos)
6. [Estado del Arte](#6-estado-del-arte)
7. [Propuesta de Solución](#7-propuesta-de-solución)
8. [Implementación Técnica](#8-implementación-técnica)
9. [Requerimientos](#9-requerimientos)
10. [Criterios de Aceptación](#10-criterios-de-aceptación)
11. [Cronograma del Proyecto](#11-cronograma-del-proyecto)
12. [Instalación y Uso](#12-instalación-y-uso)
13. [Referencias](#13-referencias)

---

## 1. Introducción

Banana Kong es un videojuego de plataformas y carrera continua (*endless runner*) desarrollado por FDG Entertainment, disponible para plataformas móviles Android e iOS. El juego presenta a un gorila que debe desplazarse por una selva tropical recolectando plátanos, esquivando obstáculos y utilizando animales de apoyo para avanzar. Su espacio de acciones reducido (salto/planeo, dash y agacharse) lo convierte en un candidato adecuado para el entrenamiento de un agente basado en aprendizaje por refuerzo.

Este proyecto propone construir un agente que perciba el juego exclusivamente a través de la pantalla y ejecute acciones simulando entradas de teclado, sin acceso a la memoria del juego ni modificación del APK. El módulo de percepción usa visión por computador (OpenCV) con detectores especializados por tipo de objeto; el módulo de decisión usa PPO (Proximal Policy Optimization) implementado con Stable-Baselines3.

---

## 2. Planteamiento del Problema

Los videojuegos comerciales son sistemas de caja negra: no exponen su estado interno mediante APIs públicas. La única información disponible para un agente externo es la imagen renderizada en pantalla. Esto genera una brecha técnica concreta: integrar captura visual, percepción computacional y ejecución de acciones en un pipeline coherente que opere en tiempo real es un problema de ingeniería no trivial, especialmente con hardware académico limitado.

**Pregunta central:** ¿Es posible diseñar e implementar, bajo restricciones académicas de tiempo y hardware, un agente autónomo basado en aprendizaje por refuerzo que aprenda a jugar Banana Kong en un emulador Android para PC, utilizando únicamente información visual y simulación de entradas de teclado, alcanzando un puntaje promedio de 5.000–6.000 puntos por partida?

---

## 3. Restricciones y Supuestos

### 3.1 Restricciones Técnicas

- **Sin acceso interno al juego:** El sistema trata Banana Kong como caja negra. No se lee ni modifica la memoria del proceso, ni se inyecta código en el emulador.
- **Captura exclusivamente visual:** Toda la información del estado proviene de capturas de pantalla con `mss`. No se usa audio, tráfico de red ni otras fuentes.
- **Acciones mediante teclado simulado:** Las interacciones se ejecutan a través de `pyautogui` simulando las teclas configuradas en BlueStacks Game Controls. No se usa ADB por problemas de latencia y conflicto con eventos táctiles.
- **Resolución fija 960×540:** Todos los detectores están calibrados para esta resolución. Cambiarla requiere recalibrar ROIs y umbrales.
- **Latencia objetivo:** El ciclo completo captura → percepción → decisión → acción debe completarse en menos de 100 ms.
- **Hardware de consumo:** Desarrollo en equipos con GPU NVIDIA de gama media. Sin clústeres ni instancias cloud.

### 3.2 Restricciones del Entorno de Juego

- **Meta de puntaje:** El agente debe alcanzar un puntaje promedio de **5.000–6.000 puntos** por episodio como criterio de éxito.
- **Restricción de mundos alternativos:** El agente **no debe entrar a mundos alternativos** accesibles mediante cuevas, zonas de agua, cielo u otros portales. Estos mundos cambian radicalmente la paleta de colores, la geometría de obstáculos y la estructura del HUD, invalidando todos los detectores calibrados para el mundo principal (selva). Esta restricción se implementa penalizando fuertemente la detección de agua (que precede a las transiciones de mundo) y limitando el ROI de percepción.
- **Mundo único:** El agente opera exclusivamente en el mundo de la selva (mundo inicial). No se contemplan otros biomas.
- **Configuración gráfica fija:** La ventana del emulador permanece en primer plano y visible durante toda la ejecución.

### 3.3 Restricciones Normativas

- El proyecto es estrictamente académico y no comercial.
- No se redistribuye el APK del juego.
- El bot opera exclusivamente en modalidad de un jugador (offline).

### 3.4 Supuestos

- Los elementos clave del juego son visualmente distinguibles con las técnicas implementadas en condiciones normales del mundo selva.
- Los colores, formas y posiciones de los elementos del HUD y del entorno son consistentes entre partidas dentro del mismo mundo.
- El juego no recibirá actualizaciones que cambien significativamente su interfaz visual durante el semestre.

---

## 4. Alcance del Proyecto

### Incluido

- Pipeline completo: captura → percepción → decisión → acción
- Detectores especializados para: Kong, barriles, bananas, agua, rocas, muros (madera y piedra), game over
- Entorno compatible con la interfaz OpenAI Gymnasium
- Entrenamiento con PPO usando Stable-Baselines3
- Reinicio automático de episodios
- Evaluación frente a política aleatoria de referencia (*baseline*)
- Documentación técnica completa

### Excluido

- Soporte para múltiples juegos o biomas distintos al mundo selva
- Detección de objetos interactivos opcionales (lianas, trampolines, guacamaya, plataformas flotantes) — se dejan para iteraciones futuras una vez consolidada la política básica de supervivencia
- Interfaz gráfica de usuario (GUI): la ejecución es por línea de comandos
- Modificación del APK, archivos del emulador o código del juego
- Generalización a múltiples resoluciones o versiones del juego

---

## 5. Objetivos

### General

Diseñar e implementar un agente autónomo basado en aprendizaje por refuerzo profundo que aprenda a jugar Banana Kong en un emulador Android para PC, utilizando exclusivamente información visual de la pantalla y simulación de teclado, alcanzando al final del semestre un puntaje promedio de 5.000–6.000 puntos por episodio, superior al de una política aleatoria de referencia.

### Específicos

1. Implementar un módulo de captura capaz de obtener fotogramas del emulador a mínimo 15 FPS con latencia individual menor a 50 ms.
2. Desarrollar detectores de visión por computador para cada tipo de objeto relevante del juego, con precisión superior al 85% en condiciones normales del mundo selva.
3. Diseñar y formalizar el entorno Gymnasium con espacio de estados, acciones y función de recompensa.
4. Entrenar al menos un agente PPO durante un mínimo de 500.000 pasos, documentando curvas de aprendizaje.
5. Evaluar el agente frente a una política aleatoria, demostrando mejora estadísticamente significativa en puntaje promedio por episodio en al menos 30 episodios.
6. Documentar el sistema completo en este repositorio con READMEs, diagramas y resultados de experimentos.

---

## 6. Estado del Arte

### 6.1 Aprendizaje por Refuerzo en Videojuegos

El trabajo de Mnih et al. (2015) con DQN demostró que una red neuronal puede aprender políticas de juego competitivas directamente desde píxeles en juegos de Atari. Schulman et al. (2017) propusieron PPO, algoritmo de gradiente de política con mayor estabilidad de entrenamiento, que es el que utilizamos en este proyecto por su buen desempeño con espacios de acción discretos pequeños y su disponibilidad en Stable-Baselines3.

### 6.2 Bots para Endless Runners

Proyectos como el bot para Subway Surfers de Yeh et al. (2021) usaron visión por computador con OpenCV para detectar obstáculos mediante segmentación por color, sin aprendizaje automático. Lograron tiempos de supervivencia superiores al jugador promedio pero con robustez limitada a condiciones de color constante. Nuestro enfoque híbrido (HSV + template matching + RL) busca mayor generalización.

### 6.3 Vacíos que Abordamos

- Escasez de implementaciones académicas reproducibles de agentes RL visuales para juegos móviles en emulador
- Ausencia de pipelines completos documentados que integren percepción clásica con RL para endless runners en plataformas de consumo
- Falta de comparación directa entre detección puramente por color vs. detección híbrida para objetos con alta variación visual

---

## 7. Propuesta de Solución

### 7.1 Arquitectura General

```
BlueStacks (960x540)
        │
        ▼
┌───────────────┐
│    mss        │  Captura de pantalla (~60 FPS)
└───────┬───────┘
        │ frame BGR
        ▼
┌───────────────┐
│  Perceptor    │  Corre todos los detectores sobre el frame
└───────┬───────┘
        │ estado (dict)
        ▼
┌───────────────┐
│  BananaKong   │  Entorno Gymnasium: convierte estado → obs vector
│  Env          │  Calcula reward, detecta terminación
└───────┬───────┘
        │ obs (24 floats)
        ▼
┌───────────────┐
│  PPO Agent    │  Stable-Baselines3: selecciona acción 0-3
│  (MlpPolicy)  │
└───────┬───────┘
        │ acción
        ▼
┌───────────────┐
│  ModuloAccio  │  pyautogui: ejecuta tecla en BlueStacks
│  nes          │
└───────────────┘
```

### 7.2 Espacio de Acciones

| ID | Acción | Tecla BlueStacks | Descripción |
|----|--------|-----------------|-------------|
| 0 | NADA | — | El juego avanza automáticamente |
| 1 | PLANEAR | W | Tap = saltar; mantener = planear |
| 2 | DASH | D | Impulso hacia adelante |
| 3 | BAJAR | S | Deslizarse hacia abajo |

**Nota sobre la implementación de controles:** Inicialmente se intentó simular el dash mediante `pyautogui.drag()`, lo que provocaba que BlueStacks registrara el inicio del drag como un tap, haciendo saltar a Kong antes del dash. La solución adoptada fue configurar el dash directamente en el **Game Controls de BlueStacks** como una tecla (`D`), eliminando por completo la necesidad de simular gestos táctiles. Lo mismo aplica para W (salto/planeo) y S (bajar). El `keyDown`/`keyUp` de `pyautogui` sobre estas teclas globales no genera el problema del tap previo.

**Planeo implícito:** El agente controla la duración del planeo eligiendo la acción PLANEAR durante múltiples steps consecutivos. Mientras seleccione PLANEAR, el módulo mantiene W presionado (`keyDown`). Al seleccionar cualquier otra acción, suelta W (`keyUp`). Esto permite que la duración del planeo emerja del comportamiento aprendido sin necesidad de una acción separada.

### 7.3 Función de Recompensa

| Evento | Recompensa |
|--------|-----------|
| Sobrevivir cada step | +0.01 |
| Banana recogida | +1.0 por banana |
| Game over | -10.0 |

**Detección de bananas recogidas:** Se lee el contador de bananas del HUD del juego. Si el valor del HUD aumenta respecto al step anterior, Kong recogió una o más bananas. Este enfoque es más confiable que medir la disminución de bananas visibles en pantalla, ya que el número de bananas visibles varía también cuando el escenario avanza y las bananas salen del ROI sin ser recogidas.

**Restricción de agua:** La detección de agua activa el flag `hay_agua = True` en la observación. Esto informa al agente que una zona peligrosa está presente, incentivando indirectamente a evitar entrar a mundos alternativos a través de zonas acuáticas.

---

## 8. Implementación Técnica

### 8.1 Módulo de Percepción — Detectores

El `Perceptor` captura el frame **una sola vez por ciclo** y lo distribuye a todos los detectores, evitando múltiples capturas costosas.

#### Estrategia de detección por tipo de objeto

La mayoría de los detectores siguen un enfoque **híbrido HSV + Template Matching**. Los detectores de agua y bananas usan solo HSV por la alta distintividad de sus colores.

**¿Por qué HSV + Template Matching y no solo uno de los dos?**

- **Solo HSV:** Insuficiente para objetos que comparten colores con el fondo (ej. muros de madera vs. troncos de árbol, ambos marrones). Da demasiados falsos positivos.
- **Solo Template Matching:** Lento sobre el frame completo y sensible a variaciones de escala no anticipadas. Sobre el frame de 960×540, buscar un template de 60×60px en múltiples escalas toma ~15ms por objeto.
- **Híbrido:** HSV reduce el frame a un conjunto pequeño de blobs candidatos (típicamente 1–5 por objeto). Template matching corre solo sobre esos recortes pequeños, siendo 10–50x más rápido y más preciso.

| Detector | Estrategia | Razón |
|----------|-----------|-------|
| Kong | HSV (marrón piel) + template multi-pose | Varias poses (correr, saltar, planear, dash) |
| Bananas | HSV (amarillo intenso) | Color muy distintivo, ROI excluye zona de Kong |
| Agua | HSV (azul/celeste) | Color único en la escena, sin template necesario |
| Barriles | HSV (marrón V alto) + template | V alto distingue interior brillante del suelo oscuro |
| Rocas | Template matching (TM_CCOEFF_NORMED) | Dos tipos: roca orgánica y roca con estructura; colores no separables con HSV del fondo |
| Muros madera | HSV (naranja S>150) + template | S alto distingue de troncos (S~100-120) |
| Muros piedra | HSV (gris rosado) + template | Rango estrecho de saturación |
| Game Over | Template matching puro sobre ROI central | Pantalla estática, muy confiable |

#### Configuración de ROIs

Los ROIs están expresados en **píxeles absolutos** calibrados para la resolución 960×540 de BlueStacks. Cambiar la resolución requiere recalibrar todos los ROIs.

```python
# Ejemplos de ROIs utilizados
ROI_KONG     = (0, 60, 420, 510)      # tercio izquierdo, excluye HUD
ROI_BARRILES = (250, 80, 900, 480)    # excluye zona de Kong
ROI_BANANAS  = (350, 60, 960, 510)    # excluye Kong y su boca
ROI_AGUA     = (0, 300, 960, 510)     # franja inferior
ROI_GAMEOVER = (200, 100, 760, 400)   # zona central de pantalla
```

#### Templates

Todos los templates se entregan como **PNG con canal alpha real** (no fondo negro). El proceso de generación fue:

1. Recortar el objeto del screenshot del juego con fondo removido (removebg.com)
2. Si el PNG resultante tiene 3 canales con fondo negro (caso común con removebg), regenerar el alpha con `cv2.threshold(gris, 8, 255, THRESH_BINARY)`
3. Guardar como RGBA con el alpha correcto

El template matching usa `TM_CCOEFF_NORMED` con la alpha mask:
```python
cv2.matchTemplate(recorte, template, cv2.TM_CCOEFF_NORMED, mask=alpha)
```
Se eligió `TM_CCOEFF_NORMED` sobre `TM_SQDIFF_NORMED` porque el máximo es la mejor coincidencia (más intuitivo), es más robusto ante variaciones de brillo, y no favorece zonas oscuras del frame cuando no hay alpha.

### 8.2 Vector de Observación

El estado del juego se convierte a un vector de **24 floats normalizados [0, 1]**:

```
[0-1]   kong_cx, kong_cy
[2-7]   barril1_cx, barril1_cy, barril2_cx, barril2_cy, barril3_cx, barril3_cy
[8-13]  banana1_cx, banana1_cy, banana2_cx, banana2_cy, banana3_cx, banana3_cy
[14-17] muro1_cx, muro1_cy, muro2_cx, muro2_cy
[18]    hay_agua (0 o 1)
[19-23] kong_pose one-hot (inicio, corriendo, saltando, paracaidas, dash)
```

Los barriles, bananas y muros se ordenan por distancia horizontal a Kong, de modo que el índice 0 siempre corresponde al más cercano (el más relevante para la decisión). Las rocas no se incluyen explícitamente en el vector de observación — el agente aprende a evitarlas mediante la penalización de game over al colisionar con ellas.

### 8.3 Módulo de Acciones

```python
# acciones.py
TECLA_SALTAR = 'w'   # configurada en BlueStacks Game Controls
TECLA_DASH   = 'd'
TECLA_BAJAR  = 's'
```

El estado `_planeando` evita enviar `keyDown` repetidos:
```python
elif accion == PLANEAR:
    if not self._planeando:
        pyautogui.keyDown(TECLA_SALTAR)
        self._planeando = True
    # Si ya planeaba: no hace nada, W sigue presionada
```

### 8.4 Entrenamiento

```python
# entrenar.py
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps":       512,
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
}
```

```bash
python entrenar.py              # desde cero
python entrenar.py --continuar  # continuar desde checkpoint
```

Los checkpoints se guardan cada 10.000 steps en `modelos/checkpoints/`. El modelo final se guarda en `modelos/banana_kong_ppo.zip`.

---

## 9. Requerimientos

### Funcionales

| ID | Requerimiento |
|----|--------------|
| RF-01 | Capturar fotogramas del emulador a mínimo 15 FPS |
| RF-02 | Detectar posición de Kong, barriles, bananas, rocas, muros y agua en cada frame |
| RF-03 | Detectar fin de episodio (game over) con máximo 1s de latencia |
| RF-04 | Reiniciar el juego automáticamente al final de cada episodio |
| RF-05 | Exponer entorno compatible con OpenAI Gymnasium (step, reset, render) |
| RF-06 | Entrenar agente PPO y guardar checkpoints periódicos |
| RF-07 | Evaluar agente entrenado y comparar con política aleatoria |
| RF-08 | Ejecutar acciones mediante teclas configuradas en BlueStacks |
| RF-09 | Registrar métricas de entrenamiento en logs para TensorBoard |

### No Funcionales

| ID | Requerimiento |
|----|--------------|
| RNF-01 | Ciclo completo captura → decisión → acción < 100 ms (90% de los casos) |
| RNF-02 | Captura sostenida ≥ 15 FPS durante sesiones de > 30 minutos |
| RNF-03 | ROIs calibrados para resolución fija 960×540 de BlueStacks 5 |
| RNF-04 | Sistema ejecutable con un solo comando desde terminal |
| RNF-05 | Código organizado en módulos independientes con docstrings |
| RNF-06 | Compatible con Windows 10/11, Python 3.9+, BlueStacks 5 |

---

## 10. Criterios de Aceptación

| ID | Criterio | Métrica |
|----|----------|---------|
| CA-01 | Captura funcional | ≥ 15 FPS durante 30 min continuas |
| CA-02 | Latencia del pipeline | < 100 ms en el 90% de los ciclos |
| CA-03 | Reinicio automático | Exitoso en ≥ 95% de los episodios |
| CA-04 | Compatibilidad Gym | Entorno ejecuta episodios completos sin errores |
| CA-05 | Entrenamiento completado | ≥ 500.000 pasos sin fallos críticos |
| CA-06 | Superación de baseline | Puntaje promedio agente > baseline en 30 episodios (t-test p < 0.05) |
| CA-07 | Meta de puntaje | Promedio ≥ 5.000 puntos por episodio tras entrenamiento completo |
| CA-08 | Restricción de mundos | El agente no entra a mundos alternativos en ≥ 90% de episodios evaluados |
| CA-09 | Repositorio documentado | README completo, instrucciones reproducibles, código comentado |

---

## 11. Cronograma del Proyecto

> **Versión del cronograma:** v2 (entrega `crono1`) — 2026-03-11

| Semanas | Fase | Actividades principales | Prototipo |
|---------|------|------------------------|-----------|
| 1–2 | Selección del videojuego | Evaluación de candidatos, criterios de selección, decisión documentada, boceto del pipeline | P1: Documento de selección y justificación de Banana Kong |
| 3–4 | Configuración del entorno | Instalación BlueStacks 960×540, configuración Game Controls (W/D/S), instalación de dependencias Python, benchmark FPS | P2: Captura funcional a ≥ 15 FPS con reporte de latencia en consola |
| 4–7 | Módulo de percepción | Preparación de templates PNG con alpha; detectores de Kong (9 poses), bananas, agua, barriles, rocas (2 tipos), muros (madera y piedra), game over; integración en clase `Perceptor`; calibración de ROIs | P3: `perceptor.py` en modo demo con bounding boxes en tiempo real sobre todos los objetos |
| 5–7 | Entorno Gymnasium | Diseño del espacio de observación (24 floats) y acciones (Discrete(4)); implementación de `BananaKongEnv`; función de recompensa con contador del HUD; reinicio automático en 3 pasos; período de gracia de 60 steps | P4: Entorno ejecutando episodios completos sin errores + `baseline.py` con 10 episodios de política aleatoria |
| 7–8 | Integración del pipeline | Integración captura → percepción → decisión → acción; medición de latencia end-to-end; corrección de cuellos de botella; validación de ROIs en partidas reales | P5: Pipeline completo corriendo 5 episodios sin intervención manual, latencia p90 < 100 ms |
| 8–13 | Entrenamiento RL con PPO | Configuración de hiperparámetros; corrida inicial 100k pasos; monitoreo en TensorBoard; ajuste de hiperparámetros; corrida completa ≥ 500k pasos con checkpoints cada 10k | P6: Checkpoint a 100k pasos con curva de reward creciente<br>P7: Modelo final `banana_kong_ppo.zip` a ≥ 500k pasos |
| 13–14 | Evaluación | Evaluación formal agente PPO (30 episodios); evaluación baseline aleatorio (30 episodios); t-test de medias; verificación restricción de mundos alternativos | P8: Informe de evaluación con tabla de puntajes, t-test y distribución (boxplot) |
| 14–15 | Optimización y robustez | Ajuste fino según fallos identificados; re-entrenamiento parcial si hay regresión; pruebas de robustez; corrección de detectores con mayor tasa de error | P9: Agente refinado con puntaje promedio ≥ 5.000 puntos |
| 15–16 | Documentación y entrega | Limpieza del repositorio; actualización del README con resultados; reporte final; grabación de demo (≥ 3 episodios); presentación | P10: Entrega final con repositorio documentado, demo en video y todos los CA verificados |

### Diagrama de Gantt

```
Sem →   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
─────────────────────────────────────────────────────────
Selección [P1]
        ████
Configuración [P2]
              ████
Percepción [P3]
                 ████████████
Entorno Gym [P4]
                    ████████
Integración [P5]
                          ████
Entrenamiento [P6→P7]
                              ████████████████
Evaluación [P8]
                                          ████
Optimización [P9]
                                              ████
Documentación [P10]
                                                  ████
─────────────────────────────────────────────────────────
Entrega crono1: semana 4 (branch crono1)
```

---

## 12. Instalación y Uso

### Requisitos

```
Python 3.9+
BlueStacks 5 (resolución 960x540)
NVIDIA GPU con CUDA (recomendado)
```

### Instalación

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
```

### Configuración de BlueStacks

En BlueStacks, configurar el **Game Controls** del juego con las siguientes teclas:

| Tecla | Acción en el juego |
|-------|-------------------|
| `W` | Saltar / Planear (tap = saltar, mantener = planear) |
| `D` | Dash (impulso hacia adelante) |
| `S` | Bajar / Deslizarse |

> **Importante:** Esta configuración reemplaza los gestos táctiles, eliminando el problema de que BlueStacks interprete el inicio de un swipe como un tap.

### Templates

Colocar los templates con alpha en la carpeta `templates/`:

```
templates/
├── kong_corriendo-bg.png
├── kong_saltando-bg.png
├── kong_paracaidas-bg.png
├── kong_dash-bg.png
├── (y otros 5 templates de poses de Kong)
├── barril-bg.png
├── roca1-bg.png
├── roca2-bg.png
├── muro_madera-bg.png
├── muro_piedra-bg.png
├── revive_texto.png
├── flecha.png
└── play_again.png
```

### Entrenamiento

```bash
# Asegurarse de que BlueStacks esté abierto con Banana Kong corriendo
python entrenar.py

# Continuar entrenamiento previo
python entrenar.py --continuar

# Ver métricas en TensorBoard
tensorboard --logdir logs/
```

### Probar detectores individualmente

```bash
python detector_kong.py
python detector_bananas.py
python detector_barriles.py
python detector_rocas.py
python detector_muros.py
python detector_agua.py
python detector_gameover.py
python perceptor.py       # todos juntos
```

---

## 13. Referencias

1. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.
2. OpenAI, "OpenAI Five," 2019. https://openai.com/five
3. O. Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning," *Nature*, vol. 575, pp. 350–354, 2019.
4. J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
5. V. Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning," *ICML*, 2016.
6. Y.-H. Yeh et al., "Automated Game Bot for Subway Surfers Using Computer Vision," *IEEE ICCE*, 2021.
7. G. Brockman et al., "OpenAI Gym," arXiv:1606.01540, 2016.
8. A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *JMLR*, vol. 22, 2021.
9. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal*, 2000.
10. ScreenInfo, "mss: An ultra-fast cross-platform multiple screenshots module," https://python-mss.readthedocs.io