import cv2
import numpy as np
import os

class ProcesadorImagenes:
    def __init__(self, ruta_db, ruta_masks, ruta_processed):
        self.ruta_db = ruta_db
        self.ruta_masks = ruta_masks
        self.ruta_processed = ruta_processed
        os.makedirs(self.ruta_masks, exist_ok=True)
        os.makedirs(self.ruta_processed, exist_ok=True)

    def redimensionar_con_bordes(self, imagen, nueva_dim=(256, 256)):
        altura, ancho = imagen.shape[:2]
        max_dim = max(altura, ancho)
        escala = nueva_dim[0] / max_dim
        nueva_altura = int(altura * escala)
        nuevo_ancho = int(ancho * escala)
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nueva_altura), interpolation=cv2.INTER_AREA)

        imagen_con_bordes = np.zeros((nueva_dim[0], nueva_dim[1], 3), dtype=np.uint8)
        y_offset = (nueva_dim[0] - nueva_altura) // 2
        x_offset = (nueva_dim[1] - nuevo_ancho) // 2
        imagen_con_bordes[y_offset:y_offset + nueva_altura, x_offset:x_offset + nuevo_ancho] = imagen_redimensionada

        return imagen_con_bordes

    def metodo_berenjenas_camotes(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gris_ecualizado = clahe.apply(gris)
        gris_suavizado = cv2.GaussianBlur(gris_ecualizado, (5, 5), 0)
        mediana_intensidad = np.median(gris_suavizado)
        bajo = int(max(0, 0.7 * mediana_intensidad))
        alto = int(min(255, 1.3 * mediana_intensidad))
        bordes = cv2.Canny(gris_suavizado, bajo, alto)
        kernel = np.ones((3, 3), np.uint8)
        bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(bordes_cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mascara = np.zeros_like(gris)
        if contornos:
            contornos = [c for c in contornos if cv2.contourArea(c) > 100]
            if contornos:
                contorno_mas_grande = max(contornos, key=cv2.contourArea)
                cv2.drawContours(mascara, [contorno_mas_grande], -1, 255, thickness=cv2.FILLED)
        return mascara

    def metodo_zanahorias_papas(self, imagen):
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        rango_bajo = (5, 50, 50)
        rango_alto = (20, 255, 255)
        mascara_hsv = cv2.inRange(hsv, rango_bajo, rango_alto)
        mascara_hsv_suavizada = cv2.GaussianBlur(mascara_hsv, (5, 5), 0)
        _, binarizada_global = cv2.threshold(mascara_hsv_suavizada, 127, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(binarizada_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mascara_final = np.zeros_like(binarizada_global)
        if contornos:
            contorno_mas_grande = max(contornos, key=cv2.contourArea)
            cv2.drawContours(mascara_final, [contorno_mas_grande], -1, 255, thickness=cv2.FILLED)
        return mascara_final

    def procesar_imagen_candidata(self, ruta_imagen):
        carpeta_candidato = os.path.dirname(ruta_imagen)
        ruta_mascara = os.path.join(carpeta_candidato, "Mask")
        ruta_procesada = os.path.join(carpeta_candidato, "Processed")

        os.makedirs(ruta_mascara, exist_ok=True)
        os.makedirs(ruta_procesada, exist_ok=True)

        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"No se pudo cargar la imagen candidata: {ruta_imagen}")
            return

        imagen = self.redimensionar_con_bordes(imagen)
        mascara_berenjenas_camotes = self.metodo_berenjenas_camotes(imagen)
        mascara_zanahorias_papas = self.metodo_zanahorias_papas(imagen)
        blancos_metodo1 = np.sum(mascara_berenjenas_camotes == 255)
        blancos_metodo2 = np.sum(mascara_zanahorias_papas == 255)
        mejor_mascara = mascara_berenjenas_camotes if blancos_metodo1 > blancos_metodo2 else mascara_zanahorias_papas

        imagen_mascara_aplicada = cv2.bitwise_and(imagen, imagen, mask=mejor_mascara)

        ruta_mascara = os.path.join(ruta_mascara, "mask_candidata.png")
        ruta_procesada = os.path.join(ruta_procesada, "processed_candidata.png")
        cv2.imwrite(ruta_mascara, mejor_mascara)
        cv2.imwrite(ruta_procesada, imagen_mascara_aplicada)

        print(f"Procesada imagen candidata: {ruta_imagen}")
