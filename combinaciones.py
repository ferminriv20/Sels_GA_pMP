import random, math
import numpy as np
from itertools import chain
from collections import Counter

class Combinaciones:
    """
    Enfoque usando estructuras nativas de Python (set de tuplas).
    Ideal para: Velocidad de inserción, eliminación y búsquedas.
    """
    def __init__(self):
        # Usamos un set para garantizar unicidad y búsquedas O(1)
        self.datos = set()

    def _normalizar(self, combinacion):
        """Convierte [6, 4, 2] en (2, 4, 6) para ignorar el orden."""
        return tuple(sorted(combinacion))

    def agregar(self, datos):
        """
        Método unificado inteligente.
        - Si recibe [1, 5, 3] -> Agrega una sola combinación.
        - Si recibe [[1, 5, 3], [2, 4, 6]] -> Itera y agrega ambas.
        """
        if not datos or len(datos) == 0:
            return

        # Verificamos el primer elemento para inferir la estructura
        primer_elemento = datos[0]

        # Si el primer elemento es una lista, tupla o array, asumimos que 'datos' es un LOTE (lista de listas)
        if isinstance(primer_elemento, (list, tuple, np.ndarray)):
            for comb in datos:
                self.datos.add(self._normalizar(comb))
        
        # Si el primer elemento es un número, asumimos que 'datos' es un INDIVIDUO único
        elif isinstance(primer_elemento, (int, float, np.number)):
            self.datos.add(self._normalizar(datos))
            
        else:
            # Fallback: Si no reconocemos el tipo, intentamos iterar (asumiendo lote genérico)
            try:
                for comb in datos:
                    self.datos.add(self._normalizar(comb))
            except Exception as e:
                print(f"Error al agregar datos: {e}")

    def generar(self, n, m, tam):
        """
        Genera una población de 'tam' individuos de manera eficiente.
        
        Args:  
            n (int): Límite superior del rango (0 a n-1).
            m (int): Tamaño de cada individuo (número de elementos).
            tam (int): Cantidad de individuos a generar.
        """
        if m > n:
            raise ValueError(f"El tamaño de la muestra (m={m}) no puede ser mayor que la población (n={n}).")

        # Calculamos el número máximo teórico de combinaciones posibles: C(n, m)
        max_combinaciones = math.comb(n, m)
        if tam > max_combinaciones:
            raise ValueError(f"La cantidad solicitada ({tam}) excede el número máximo de combinaciones posibles ({max_combinaciones}) para n={n} y m={m}.")

        count = 0
        while count < tam:
            # Generamos una tupla ordenada de m elementos únicos elegidos de range(n)
            # random.sample es muy eficiente para esto.
            comb = tuple(sorted(random.sample(range(n), m)))
            
            # Solo agregamos si es nueva para garantizar el tamaño exacto solicitado
            if comb not in self.datos:
                self.datos.add(comb)
                count += 1

    def existe(self, combinacion):
        return self._normalizar(combinacion) in self.datos

    def eliminar(self, combinacion):
        norm = self._normalizar(combinacion)
        self.datos.discard(norm)  # discard no lanza error si no existe

    def muestra(self, n):
        """Devuelve n elementos aleatorios."""
        if n > len(self.datos):
            raise ValueError("El tamaño de la muestra excede el total de datos.")
        return random.sample(list(self.datos), n)

    def total_elementos(self):
        return len(self.datos)
    
    def frecuencia(self):
        """Ejemplo de cálculo de totales: Frecuencia de cada número individual."""
        # Usamos chain.from_iterable para evitar crear una lista intermedia
        return Counter(chain.from_iterable(self.datos))




if __name__ == "__main__":
    print("--- Tests ---")
    poblacion = Combinaciones()
    
    # EJEMPLO 1: Agregar un LOTE (lista de listas) usando el mismo método

    # lote = [
    #     [1, 5, 3], 
    #     [3, 1, 5], # Duplicado
    #     [2, 4, 6],
    #     [12, 20, 30],
    #     [1, 21, 3], 
    #     [11, 2, 3],
    #     [1, 8, 27],
    # ]

    poblacion.generar(n=30, m=3, tam=10)
    
    """
    # EJEMPLO 2: Agregar un INDIVIDUO único usando el mismo método
    unico = [7, 8, 9]
    print(f"Agregando individuo único: {unico}")
    poblacion.agregar(unico)
    
    print(f"Total elementos únicos: {poblacion.total_elementos()}") # Debería ser 4
    print(f"¿Existe [6, 4, 2]? {poblacion.existe([6, 4, 2])}")       
    print("poblacion.datos:", poblacion.datos)
    print("Muestra aleatoria:", poblacion.muestra(5))
    print("Frecuencias:", poblacion.frecuencia())
"""