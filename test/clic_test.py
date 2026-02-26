import pyautogui
import time
import pygetwindow as gw

print("=== PRUEBA DE CONTROL ===")
print("1. Asegúrate que Banana Kong esté abierto en BlueStacks")
print("2. Pon el juego en pausa o en un lugar seguro")
print("3. Este script hará 5 clics en el centro de la pantalla")
print("\nComenzando en 3 segundos...")
time.sleep(3)

# Buscar BlueStacks
ventanas = gw.getWindowsWithTitle('BlueStacks')
if ventanas:
    bluestacks = ventanas[0]
    
    # Calcular centro de la ventana
    centro_x = bluestacks.left + (bluestacks.width // 2)
    centro_y = bluestacks.top + (bluestacks.height // 2)
    
    # Activar la ventana
    if bluestacks.isMinimized:
        bluestacks.restore()
    bluestacks.activate()
    
    print(f"Ventana activada. Centro: ({centro_x}, {centro_y})")
    time.sleep(1)
    
    # Hacer clics
    for i in range(5):
        pyautogui.click(centro_x, centro_y)
        print(f"Clic {i+1} enviado")
        time.sleep(0.5)
else:
    print("No encontré BlueStacks")

print("\nPrueba completada. ¿El personaje saltó?")