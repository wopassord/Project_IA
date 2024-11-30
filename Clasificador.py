import os
from Procesador_Imagenes import ProcesadorImagenes
from Parametrizador import Parametrizador
from Kmeans import KMeans

class Clasificador:
    def __init__(self, procesador, parametrizador, kmeans):
        self.procesador = procesador
        self.parametrizador = parametrizador
        self.kmeans = kmeans

    def ejecutar(self):
        while True:
            print("1. Procesar carpeta completa")
            print("2. Procesar imagen candidata")
            print("3. Clasificar imagen candidata")
            print("4. Clasificar imágenes (K-Means)")
            print("5. Salir")
            opcion = input("Seleccione una opción: ")

            if opcion == "1":
                self.procesar_carpeta_completa()
            elif opcion == "2":
                ruta_candidato = input("Ingrese la ruta de la carpeta 'Candidato': ")
                self.procesar_imagen_candidata(ruta_candidato)
            elif opcion == "3":
                ruta_candidato = input("Ingrese la ruta de la carpeta 'Candidato': ")
                self.clasificar_imagen_candidata(ruta_candidato)
            elif opcion == "4":
                self.kmeans.cargar_datos()
                clusters = self.kmeans.ejecutar_kmeans()
                if clusters is not None:
                    print("Clasificación completada y graficada.")
            elif opcion == "5":
                print("Saliendo del programa.")
                break
            else:
                print("Opción no válida.")

    def procesar_carpeta_completa(self):
        self.procesador.procesar_carpeta()
        self.parametrizador.procesar_parametrizacion()
        print("Procesamiento de carpeta completo. Imágenes y CSV generados.")

    def procesar_imagen_candidata(self, ruta_candidato):
        ruta_imagen_candidata = os.path.join(ruta_candidato, "candidata.jpg")
        if not os.path.exists(ruta_imagen_candidata):
            print(f"No se encontró la imagen candidata en: {ruta_imagen_candidata}")
            return

        self.procesador.procesar_imagen_candidata(ruta_imagen_candidata)
        self.parametrizador.procesar_candidata(ruta_imagen_candidata)
        print("Procesamiento de la imagen candidata completado.")

    def clasificar_imagen_candidata(self, ruta_candidato):
        self.kmeans.clasificar_imagen_candidata(ruta_candidato)
        print("Clasificación de la imagen candidata completada.")


if __name__ == "__main__":
    ruta_carpeta_db = 'C:\\Users\\berni\\Desktop\\ProyectoIA POO 2\\DB\\Crudas'
    ruta_carpeta_masks = 'C:\\Users\\berni\\Desktop\\ProyectoIA POO 2\\DB\\Masks'
    ruta_carpeta_processed = 'C:\\Users\\berni\\Desktop\\ProyectoIA POO 2\\DB\\Processed'
    ruta_carpeta_salida = 'C:\\Users\\berni\\Desktop\\ProyectoIA POO 2\\DB'
    ruta_csv_rgb = os.path.join(ruta_carpeta_salida, 'valores_rgb_promedio.csv')
    ruta_csv_hu = os.path.join(ruta_carpeta_salida, 'momentos_hu_escalados.csv')

    procesador = ProcesadorImagenes(ruta_carpeta_db, ruta_carpeta_masks, ruta_carpeta_processed)
    parametrizador = Parametrizador(ruta_carpeta_masks, ruta_carpeta_processed, ruta_carpeta_salida)
    kmeans = KMeans(ruta_csv_rgb, ruta_csv_hu)

    clasificador = Clasificador(procesador, parametrizador, kmeans)
    clasificador.ejecutar()
