"""
Este script lee instancias del pMP en formato OR-Library desde archivos de texto,
genera la matriz de costos completa usando el algoritmo de Floyd_Warshall.
Luego guarda estas matrices en archivos .npy para su uso posterior.
Referencia del formato OR-Library: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html
"""
import numpy as np
from pathlib import Path

def read_pmed(ruta_archivo: str) -> np.ndarray:
    """
    Lee una instancia de p-mediana en formato OR-Library 
    y devuelve la matriz de costos completa. 
    
    """
    INF = np.inf

    with open(ruta_archivo, "r") as f:
        # 1. Leer n (nodos), m (aristas), p (número de medianas)
        primera_linea = f.readline().split()
        n, m, p = map(int, primera_linea)

        # 2. Inicializar matriz de costos:
        #    - costo[i, i] = 0
        #    - costo[i, j] = INF para i != j
        costo = np.full((n, n), INF, dtype=float)
        np.fill_diagonal(costo, 0.0)

        # 3. Leer aristas (i, j, c)
        #    Nodos numerados 1..n en el archivo -> 0..n-1 en Python.
        #    Si se repite una arista, se conserva la ÚLTIMA leída.
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            i, j, c = map(int, linea.split())
            i -= 1
            j -= 1
            costo[i, j] = c
            costo[j, i] = c  

    # 4. Algoritmo de Floyd–Warshall usando broadcasting en NumPy
    for k in range(n):
        # costo[:, [k]] es una columna (n x 1)
        # costo[[k], :] es una fila (1 x n)
        # Se suman por broadcasting para obtener todos los caminos i->k->j
        costo = np.minimum(costo, costo[:, [k]] + costo[[k], :])

    return costo


def generar_datasets_pmedian(carpeta_entrada: str = "pmed", carpeta_salida: str = "datasets") -> None:
    """
    Lee todos los archivos .txt de la carpeta_entrada,
    genera la matriz de costos de cada uno y la guarda en formato .npy
    dentro de carpeta_salida.
    """
    carpeta_in = Path(carpeta_entrada)
    carpeta_out = Path(carpeta_salida)

    # Crear carpeta de salida si no existe
    carpeta_out.mkdir(parents=True, exist_ok=True)

    # Buscar todos los .txt dentro de la carpeta pmed
    archivos_txt = sorted(carpeta_in.glob("*.txt"))

    if not archivos_txt:
        print(f"No se encontraron archivos .txt en {carpeta_in.resolve()}")
        return

    for archivo in archivos_txt:
        print(f"Procesando {archivo.name} ...")
        matriz_costos = read_pmed(str(archivo))

        # Nombre de salida: pmed1.npy, pmed2.npy, etc.
        nombre_salida = archivo.stem + ".npy"
        ruta_salida = carpeta_out / nombre_salida

        np.save(ruta_salida, matriz_costos)
        print(f"  -> guardado en {ruta_salida.resolve()}")

    print("\nListo. Todas las matrices fueron generadas y guardadas en", carpeta_out.resolve())


if __name__ == "__main__":

    generar_datasets_pmedian("pmed", "datasets")