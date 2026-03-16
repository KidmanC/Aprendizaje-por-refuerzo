"""
test_reward_bananas.py — Verifica la lógica de colisión para el reward de bananas.
"""

import cv2
import numpy as np
import time
import pygetwindow as gw
from mss import mss

from deteccion.detector_kong    import DetectorKong
from deteccion.detector_bananas import DetectorBananas

MARGEN_KONG = 10


def hay_colision(kong_rect, banana_rect):
    kx, ky, kw, kh = kong_rect
    bx, by, bw, bh = banana_rect
    return (kx < bx + bw and kx + kw > bx and
            ky < by + bh and ky + kh > by)


def capturar_frame(sct, ventana):
    monitor = {
        "top":    ventana.top,
        "left":   ventana.left,
        "width":  ventana.width,
        "height": ventana.height,
    }
    screenshot = sct.grab(monitor)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)


def main():
    ventanas = gw.getWindowsWithTitle("BlueStacks")
    if not ventanas:
        print("BlueStacks no encontrado")
        return
    ventana = ventanas[0]
    sct = mss()

    det_kong    = DetectorKong()
    det_bananas = DetectorBananas()

    contador = 0
    pico_colisiones = 0

    cv2.namedWindow("Test Reward Bananas")
    print("=== TEST REWARD BANANAS ===")
    print("Presiona 'q' para salir")
    time.sleep(2)

    while True:
        frame = capturar_frame(sct, ventana)

        _, _, _, kong_rect, _  = det_kong.detectar_kong(frame)
        _, _, _, rects_bananas = det_bananas.detectar_bananas(frame)

        frame_vis = frame.copy()
        bananas_este_frame = 0

        kong_rect_expandido = None
        if kong_rect is not None:
            kx, ky, kw, kh = kong_rect
            kx -= MARGEN_KONG
            ky -= MARGEN_KONG
            kw += MARGEN_KONG * 2
            kh += MARGEN_KONG * 2
            kong_rect_expandido = (kx, ky, kw, kh)
            ox, oy, ow, oh = kong_rect
            cv2.rectangle(frame_vis, (ox, oy), (ox+ow, oy+oh), (0, 165, 255), 1)
            cv2.rectangle(frame_vis, (kx, ky), (kx+kw, ky+kh), (0, 255, 255), 2)

        colisiones_ahora = 0
        for rect in rects_bananas:
            bx, by, bw, bh = rect
            colision = kong_rect_expandido is not None and hay_colision(kong_rect_expandido, rect)
            if colision:
                colisiones_ahora += 1
                cv2.rectangle(frame_vis, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
                cv2.putText(frame_vis, "col", (bx, by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                cv2.rectangle(frame_vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)

        # Actualizar pico
        if colisiones_ahora > pico_colisiones:
            pico_colisiones = colisiones_ahora

        # Cuando las colisiones llegan a 0, contabilizar el pico
        if colisiones_ahora == 0 and pico_colisiones > 0:
            bananas_este_frame = pico_colisiones
            contador += bananas_este_frame
            print(f"  +{bananas_este_frame} banana(s) | total={contador}")
            pico_colisiones = 0

        cv2.rectangle(frame_vis, (5, 5), (300, 90), (0, 0, 0), -1)
        cv2.putText(frame_vis, f"Bananas recogidas: {contador}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_vis, f"Colisiones activas: {colisiones_ahora}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_vis, f"Pico: {pico_colisiones}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Test Reward Bananas", frame_vis)
        cv2.moveWindow("Test Reward Bananas", 100, 100)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"\nTotal bananas recogidas: {contador}")


if __name__ == "__main__":
    main()