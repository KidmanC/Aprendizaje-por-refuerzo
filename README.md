# Bot Autónomo para Videojuegos usando Visión por Computador y Aprendizaje por Refuerzo

## Introducción

Los videojuegos modernos presentan entornos complejos y dinámicos que requieren toma de decisiones en tiempo real. Tradicionalmente, la automatización de tareas en videojuegos ha dependido de acceso interno al motor del juego o de la modificación de su memoria, lo que no siempre es posible ni ético. Este proyecto aborda el desafío de crear un agente autónomo capaz de jugar un videojuego utilizando únicamente información visual de la pantalla, simulando la forma en que un humano percibe y actúa en el juego.

El avance de técnicas de visión por computador y aprendizaje automático, especialmente el aprendizaje por refuerzo profundo (Deep Reinforcement Learning), ha permitido desarrollar agentes que aprenden estrategias complejas directamente de píxeles. Proyectos emblemáticos como AlphaGo, OpenAI Five y los agentes de DeepMind para juegos de Atari han demostrado el potencial de estas tecnologías. Sin embargo, implementar un sistema completo que integre captura de pantalla, percepción visual y toma de decisiones en un entorno de juego comercial presenta desafíos significativos de ingeniería.

Este proyecto se desarrolla en el marco del curso de Proyecto Final de Ingeniería de Sistemas, siguiendo lineamientos ABET para el diseño de soluciones tecnológicas con restricciones realistas de tiempo, recursos, calidad y operación. Se propone un enfoque de diseño que abarca desde la captura eficiente de la pantalla hasta la ejecución de acciones mediante simulación de controles, con dos implementaciones alternativas del módulo de decisión: un sistema basado en reglas (Grupo 1) y un agente de aprendizaje por refuerzo (Grupo 2).

El resultado esperado es un prototipo funcional que demuestre la viabilidad de crear bots autónomos para videojuegos utilizando únicamente información visual, con aplicaciones potenciales en pruebas automatizadas, generación de datos de entrenamiento y estudios de inteligencia artificial en entornos lúdicos.

---

## Planteamiento del problema

### Contexto

Los videojuegos son sistemas interactivos complejos diseñados para ser jugados por humanos. Desde la perspectiva de un programa automatizado, presentan múltiples barreras de entrada:

1. **Naturaleza de caja negra:** La mayoría de los juegos comerciales no exponen su estado interno (posición de personajes, valores de variables, eventos) a través de APIs públicas.
2. **Interfaz exclusivamente visual:** La única forma de obtener información del juego es a través de los píxeles mostrados en pantalla.
3. **Interacción limitada a periféricos:** Las acciones deben ejecutarse simulando entradas de teclado, mouse o control, sin acceso directo a la lógica del juego.
4. **Restricciones de tiempo real:** El juego continúa avanzando mientras el bot percibe y decide, requiriendo procesamiento en latencias compatibles con la experiencia de juego (típicamente <100ms).
5. **Protecciones anti-cheat:** Muchos juegos implementan mecanismos para detectar automatización, lo que limita las opciones de juegos objetivo.

### Problema específico

¿Es posible diseñar e implementar un sistema autónomo que juegue un videojuego en tiempo real utilizando únicamente información visual de la pantalla y simulación de controles, logrando un desempeño cuantificable según métricas definidas, bajo restricciones realistas de ingeniería?

Este problema se descompone en varios subproblemas:

- **Captura eficiente:** Obtener imágenes de la pantalla con la menor latencia posible y alta tasa de fotogramas.
- **Percepción visual:** Extraer información estructurada del juego a partir de los píxeles (detección de elementos del HUD, localización de personajes y enemigos, identificación de eventos).
- **Toma de decisiones:** Generar secuencias de acciones que maximicen el desempeño en el juego, ya sea mediante lógica explícita (reglas) o aprendizaje (RL).
- **Ejecución de acciones:** Traducir las decisiones en comandos simulados de teclado, mouse o control, con precisión y sincronización.
- **Integración:** Combinar todos los módulos en un pipeline coherente que opere en tiempo real.

### Preguntas de investigación

1. ¿Qué técnicas de visión por computador (clásicas y basadas en deep learning) son más adecuadas para extraer información de juego en tiempo real?
2. ¿Qué arquitectura de agente de aprendizaje por refuerzo permite aprender políticas efectivas directamente de píxeles en el juego seleccionado?
3. ¿Cómo se comparan en desempeño, robustez y eficiencia los enfoques basados en reglas vs. aprendizaje por refuerzo?
4. ¿Qué latencia máxima es tolerable para que el bot juegue efectivamente sin desventaja respecto a un humano?

---

## Restricciones y supuestos de diseño

### Restricciones técnicas

1. **Sin acceso interno al juego:** El sistema tratará el juego como una caja negra, sin leer ni modificar su memoria, variables internas o archivos de configuración.
2. **Sin modificación del cliente:** No se alterarán los archivos del juego ni se inyectará código en su proceso.
3. **Captura exclusivamente visual:** Toda la información del estado del juego provendrá de la captura de pantalla, sin usar otras fuentes (audio, redes, APIs).
4. **Acción mediante periféricos simulados:** Las acciones se ejecutarán simulando entradas de teclado, mouse o control, a través de librerías estándar.
5. **Tiempo real:** El sistema completo (captura → percepción → decisión → acción) debe operar con latencia suficiente para mantener el ritmo del juego (objetivo: <100ms por ciclo).
6. **Recursos computacionales limitados:** El desarrollo y pruebas se realizarán en hardware de consumo estándar (laptops/desktops con GPU opcional), sin acceso a clústeres de alto rendimiento.

### Restricciones del entorno de juego

1. **Juego offline o single-player:** Para evitar complicaciones con anti-cheat y conexiones de red, el juego objetivo debe ser jugable sin conexión o en modo un jugador.
2. **Sin protección anti-cheat agresiva:** Se seleccionará un juego que no implemente mecanismos que detecten y bloqueen la automatización de entradas (ej. juegos offline, emuladores, juegos open-source).
3. **Configuración fija:** El juego se ejecutará con resolución, calidad gráfica y otros parámetros constantes para garantizar reproducibilidad.
4. **Ventana en primer plano:** Se asume que la ventana del juego permanecerá en primer plano y visible durante la ejecución del bot.

### Restricciones del proyecto

1. **Duración:** El proyecto se desarrollará en un ciclo académico (16 semanas), con entregables parciales.
2. **Equipo:** Dos grupos trabajando en paralelo, con coordinación para definir el juego objetivo y las interfaces entre módulos.
3. **Enfoque en diseño de ingeniería:** Se prioriza la aplicación de estándares de ingeniería de software: documentación de requerimientos, diseño de arquitectura, pruebas unitarias y de integración, control de versiones.
4. **Evaluación cuantitativa:** El éxito se medirá con métricas objetivas (puntaje, tiempo de supervivencia, tasa de victoria), no solo con demostraciones cualitativas.

### Supuestos

1. **Visibilidad suficiente:** El juego proporciona información visual clara y consistente que puede ser interpretada automáticamente (HUD legible, elementos distinguibles).
2. **Consistencia visual:** Los elementos visuales del juego (colores, formas, posiciones relativas) son suficientemente estables para permitir detección por template matching o modelos entrenados.
3. **Sin necesidad de multitarea:** El bot se enfocará en un solo objetivo medible a la vez (ej. completar un nivel, sobrevivir cierto tiempo), no en la totalidad del juego.
4. **Acciones discretas:** Las acciones del bot se modelarán como comandos discretos (presionar tecla X, hacer clic en coordenada Y), asumiendo que el juego responde consistentemente.
5. **Reproducibilidad controlada:** Se utilizarán semillas aleatorias y configuraciones fijas para permitir comparaciones justas entre experimentos.

---

## Alcance

### Incluye

1. **Selección del juego objetivo (en conjunto con ambos grupos):**
   - Evaluación de candidatos según criterios: accesibilidad visual, ausencia de anti-cheat, complejidad adecuada, interés académico.
   - Definición de métricas de éxito específicas para el juego elegido.

2. **Módulo de captura de pantalla:**
   - Implementación de captura eficiente con librerías optimizadas (mss, dxcam).
   - Configuración de región de interés (ROI) para reducir procesamiento.
   - Medición y optimización de FPS y latencia.

3. **Módulo de percepción visual (Grupo 2):**
   - Preprocesamiento de imágenes (escalado, normalización, filtrado).
   - Detección de elementos del HUD: vida, puntaje, tiempo, munición (usando OCR, detección de colores, template matching).
   - Localización de personajes, enemigos y obstáculos (usando OpenCV y/o YOLO).
   - Seguimiento de objetos en movimiento (opcional, según necesidad del juego).
   - Generación de representación estructurada del estado para el agente RL.

4. **Módulo de decisión (Grupo 2 - Aprendizaje por Refuerzo):**
   - Diseño del entorno (environment) compatible con interfaz Gym.
   - Implementación de agente RL (DQN, PPO u otro algoritmo adecuado).
   - Entrenamiento y ajuste de hiperparámetros.
   - Evaluación de políticas aprendidas.

5. **Módulo de acción:**
   - Simulación de entradas de teclado y mouse (pydirectinput, pyautogui).
   - Sincronización de acciones con el estado del juego.
   - Manejo de combinaciones de teclas y secuencias.

6. **Arquitectura y documentación:**
   - Diagramas de componentes y flujo de datos.
   - Especificación de interfaces entre módulos.
   - Plan de pruebas unitarias y de integración.
   - Documentación de experimentos y resultados.

7. **Entregables:**
   - Prototipo funcional demostrable en el juego seleccionado.
   - Reporte técnico con: introducción, marco teórico, diseño, implementación, experimentos, resultados, conclusiones y trabajo futuro.
   - Repositorio con código fuente documentado y READMEs por módulo.

### No incluye

1. **Juegos multijugador online:** Por restricciones de anti-cheat y latencia de red.
2. **Modificación del cliente del juego:** No se alterarán archivos binarios, memoria, ni se inyectará código.
3. **Acceso a memoria interna:** Toda la información provendrá exclusivamente de la pantalla.
4. **Soporte para múltiples juegos:** El proyecto se enfocará en un solo juego objetivo.
5. **Juegos con requisitos gráficos extremos:** Se priorizarán juegos que puedan ejecutarse fluidamente en hardware disponible.
6. **Interfaz gráfica de usuario (GUI) para el bot:** El bot se ejecutará por línea de comandos o scripts.
7. **Despliegue en producción:** El proyecto es un prototipo académico, no un producto comercial.
8. **Juegos con protección anti-cheat activa:** Se excluyen juegos que bloqueen activamente la automatización (ej. Valorant, Fortnite, etc.).

### Limitaciones conocidas

- El bot puede no generalizar a cambios significativos en la interfaz del juego (actualizaciones, cambios de resolución, skins).
- El rendimiento del agente RL dependerá críticamente de la calidad de la función de recompensa y del tiempo de entrenamiento disponible.
- La latencia acumulada del pipeline puede afectar el desempeño en juegos que requieren reflejos muy rápidos.
- Algunos elementos visuales pueden ser difíciles de detectar consistentemente (texto pequeño, efectos de partículas, transparencias).

---
