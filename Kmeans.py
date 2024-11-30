import cv2
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KMeans:
    def __init__(self, ruta_rgb, ruta_hu, k=4, max_iter=5):
        self.ruta_rgb = ruta_rgb
        self.ruta_hu = ruta_hu
        self.k = k
        self.max_iter = max_iter
        self.datos = None
        self.centroides = None
        self.nombres_grupos = [f"Grupo {i+1}" for i in range(k)]  # Inicialmente genéricos

    def cargar_datos(self):
        # Leer archivos CSV
        df_rgb = pd.read_csv(self.ruta_rgb)
        df_hu = pd.read_csv(self.ruta_hu)

        valores_rgb = df_rgb[['Promedio R', 'Promedio G', 'Promedio B']].to_numpy()
        momentos_hu = df_hu.iloc[:, 1:].to_numpy()  # Tomar todas las columnas excepto los nombres

        # Preparar dataset utilizando Hu_1, Promedio G y Promedio B
        self.datos = np.array([
            momentos_hu[:, 0],  # Hu_1
            valores_rgb[:, 1] / 255,  # Promedio G normalizado
            valores_rgb[:, 2] / 255   # Promedio B normalizado
        ]).T

    @staticmethod
    def calcular_distancia(punto, centroides):
        return np.linalg.norm(punto - centroides, axis=1)

    def inicializar_centroides(self):
        indices = random.sample(range(len(self.datos)), self.k)
        self.centroides = self.datos[indices]

    def asignar_clusters(self):
        clusters = []
        for punto in self.datos:
            distancias = self.calcular_distancia(punto, self.centroides)
            clusters.append(np.argmin(distancias))
        return np.array(clusters)

    def actualizar_centroides(self, clusters):
        nuevos_centroides = []
        for i in range(self.k):
            puntos_cluster = self.datos[clusters == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides.append(np.mean(puntos_cluster, axis=0))
            else:
                nuevos_centroides.append(self.datos[random.randint(0, len(self.datos) - 1)])
        self.centroides = np.array(nuevos_centroides)

    def verificar_balance(self, clusters):
        tamanos = [np.sum(clusters == i) for i in range(self.k)]
        return all(t > 0 for t in tamanos)

    def graficar_kmeans(self, clusters, iteracion):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        colores_clusters = ['r', 'g', 'b', 'y']

        for i in range(len(self.centroides)):
            puntos = self.datos[clusters == i]
            ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], c=colores_clusters[i], label=f'{self.nombres_grupos[i]}')

        ax.scatter(self.centroides[:, 0], self.centroides[:, 1], self.centroides[:, 2], c='k', s=100, marker='X', label='Centroides')

        ax.set_xlabel('Hu_1')
        ax.set_ylabel('Promedio G')
        ax.set_zlabel('Promedio B')
        ax.set_title(f'Iteración {iteracion}')
        ax.legend()
        plt.show()

    def ejecutar_kmeans(self):
        for intento in range(10):  # Intentar hasta 10 veces
            self.inicializar_centroides()
            for iteracion in range(self.max_iter):
                clusters = self.asignar_clusters()
                self.graficar_kmeans(clusters, iteracion + 1)
                self.actualizar_centroides(clusters)

            if self.verificar_balance(clusters):
                print("Clasificación completada con éxito.")
                self.asignar_nombres_grupos()
                return clusters

        print("No se logró una clasificación balanceada.")
        return None

    def asignar_nombres_grupos(self):
        print("Asignación de nombres a los grupos:")
        for i in range(self.k):
            nombre = input(f"Ingrese el nombre para el Grupo {i+1} (actual: {self.nombres_grupos[i]}): ")
            if nombre:
                self.nombres_grupos[i] = nombre

    def clasificar_imagen_candidata(self, carpeta_candidato):
        ruta_rgb_candidato = os.path.join(carpeta_candidato, 'valores_rgb_candidata.csv')
        ruta_hu_candidato = os.path.join(carpeta_candidato, 'momentos_hu_candidata.csv')

        if not os.path.exists(ruta_rgb_candidato) or not os.path.exists(ruta_hu_candidato):
            print("No se encontraron los archivos CSV del candidato en la carpeta especificada.")
            return

        # Cargar los datos del candidato
        df_rgb_candidato = pd.read_csv(ruta_rgb_candidato)
        df_hu_candidato = pd.read_csv(ruta_hu_candidato)

        # Extraer valores de Hu_1, Promedio G y Promedio B
        hu1 = df_hu_candidato.iloc[0, 1]  # Primera fila, columna Hu_1
        g = df_rgb_candidato.iloc[0, 2]  # Primera fila, columna Promedio G
        b = df_rgb_candidato.iloc[0, 3]  # Primera fila, columna Promedio B

        punto_candidato = np.array([hu1, g / 255, b / 255])
        distancias = self.calcular_distancia(punto_candidato, self.centroides)
        grupo = np.argmin(distancias)
        print(f"La imagen candidata pertenece a: {self.nombres_grupos[grupo]}.")
