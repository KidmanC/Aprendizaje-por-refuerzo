"""
evaluar.py — Evaluación del agente entrenado vs baseline aleatorio
Uso:
    python -m entrenamiento.evaluar                    # evaluar modelo guardado
    python -m entrenamiento.evaluar --baseline         # solo baseline aleatorio
    python -m entrenamiento.evaluar --ambos            # agente + baseline + comparación
"""

import argparse
import os
import time
import numpy as np
from scipy import stats

from stable_baselines3 import PPO
from entorno.entorno import BananaKongEnv

RUTA_MODELO  = "modelos/banana_kong_ppo"
N_EPISODIOS  = 30


def evaluar_agente(env, modelo, n_episodios, etiqueta="Agente"):
    rewards   = []
    duraciones = []
    bananas   = []

    print(f"\n=== {etiqueta} ({n_episodios} episodios) ===")
    for ep in range(n_episodios):
        obs, _ = env.reset()
        reward_total = 0
        steps        = 0
        bananas_ep   = 0

        while True:
            if modelo is not None:
                accion, _ = modelo.predict(obs, deterministic=True)
            else:
                accion = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(int(accion))
            reward_total += reward
            steps        += 1
            bananas_ep   += info.get("bananas", 0)

            if terminated or truncated:
                break

        rewards.append(reward_total)
        duraciones.append(steps)
        bananas.append(bananas_ep)
        print(f"  Ep {ep+1:2d}: reward={reward_total:+7.2f}  steps={steps:4d}  bananas={bananas_ep}")

    print(f"\n  Reward  — media={np.mean(rewards):+.2f}  std={np.std(rewards):.2f}  min={np.min(rewards):+.2f}  max={np.max(rewards):+.2f}")
    print(f"  Steps   — media={np.mean(duraciones):.1f}  std={np.std(duraciones):.1f}")
    print(f"  Bananas — media={np.mean(bananas):.1f}  total={sum(bananas)}")

    return rewards, duraciones, bananas


def comparar(rewards_agente, rewards_baseline):
    print("\n=== COMPARACIÓN ESTADÍSTICA ===")
    media_a = np.mean(rewards_agente)
    media_b = np.mean(rewards_baseline)
    mejora  = media_a - media_b

    t_stat, p_valor = stats.ttest_ind(rewards_agente, rewards_baseline)

    print(f"  Agente:   {media_a:+.2f}")
    print(f"  Baseline: {media_b:+.2f}")
    print(f"  Mejora:   {mejora:+.2f}")
    print(f"  t-stat:   {t_stat:.3f}")
    print(f"  p-valor:  {p_valor:.4f}")

    if p_valor < 0.05 and media_a > media_b:
        print("\n  ✅ El agente supera el baseline (p < 0.05) — objetivo del proyecto CUMPLIDO")
    elif media_a > media_b:
        print(f"\n  ⚠️  El agente tiene mejor reward pero no es estadísticamente significativo (p={p_valor:.3f})")
        print("      Necesitás más episodios de evaluación o más entrenamiento.")
    else:
        print("\n  ❌ El agente NO supera el baseline — necesita más entrenamiento.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Solo evaluar baseline aleatorio")
    parser.add_argument("--ambos",    action="store_true", help="Evaluar agente + baseline y comparar")
    parser.add_argument("--n",        type=int, default=N_EPISODIOS, help="Número de episodios")
    parser.add_argument("--modelo",   type=str, default=RUTA_MODELO, help="Ruta al modelo")
    args = parser.parse_args()

    print("=== EVALUACIÓN BANANA KONG ===")
    print("Asegurate de que BlueStacks esté abierto y el juego corriendo")
    time.sleep(3)

    env = BananaKongEnv()

    if args.baseline:
        evaluar_agente(env, None, args.n, etiqueta="Baseline aleatorio")

    elif args.ambos:
        ruta = args.modelo + ".zip" if not args.modelo.endswith(".zip") else args.modelo
        if not os.path.exists(ruta):
            print(f"❌ No se encontró modelo en {ruta}")
            return
        modelo = PPO.load(args.modelo, env=env)
        rewards_a, _, _ = evaluar_agente(env, modelo, args.n, etiqueta="Agente PPO")
        rewards_b, _, _ = evaluar_agente(env, None,   args.n, etiqueta="Baseline aleatorio")
        comparar(rewards_a, rewards_b)

    else:
        ruta = args.modelo + ".zip" if not args.modelo.endswith(".zip") else args.modelo
        if not os.path.exists(ruta):
            print(f"❌ No se encontró modelo en {ruta}")
            print(f"   Buscando checkpoints en modelos/checkpoints/...")
            checkpoints = sorted([
                f for f in os.listdir("modelos/checkpoints")
                if f.endswith(".zip")
            ]) if os.path.exists("modelos/checkpoints") else []
            if checkpoints:
                print(f"   Último checkpoint: {checkpoints[-1]}")
                args.modelo = os.path.join("modelos/checkpoints", checkpoints[-1][:-4])
            else:
                print("   No hay checkpoints disponibles.")
                return
        modelo = PPO.load(args.modelo, env=env)
        evaluar_agente(env, modelo, args.n, etiqueta="Agente PPO")

    env.close()


if __name__ == "__main__":
    main()