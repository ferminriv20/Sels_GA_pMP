import numpy as  np 
from numba import jit, njit, prange
from GeneticAlgorithm_V2 import AlgoritmoGenetico
from math import comb, ceil, log
from combinaciones import Combinaciones
"""
Este módulo contiene todas las funciones para implementar el algoritmo genético 
para el problema p-mediana (pMP). Se definen las funciones de evaluación, generación de población inicial, métodos de 
selección, cruzamiento, mutación, heurística de búsqueda local y criterio de parada. 
Cada función está documentada con sus argumentos y valores de retorno.

El diseño del GA se basa en la codificación clásica, donde cada individuo representa 
un conjunto de instalaciones seleccionadas. No se permiten valores repetidos en un 
mismo individuo, y el tamaño de cada individuo es fijo (p).
"""

def evaluar_poblacion(poblacion: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Cálcula el fitness de cada individuo en una población 
    Args:
    poblacion : np.ndarray
        Matriz de tamaño (pop_size, p), donde cada fila contiene los índices
        de las instalaciones seleccionadas (0..n-1) para un individuo.
    cost_matrix : np.ndarray
        Matriz de distancias/costos de tamaño (n, n) clients = facilities.
        
    Returns:
    costos : np.ndarray
        Vector de tamaño (pop_size,) con el costo total de cada individuo.
    
    """
    # cost_matrix tiene shape (n, n)
    # poblacion tiene shape (pop_size, p)
    # Esto genera una matriz de shape (n, pop_size, p)
    # Para cada cliente (n),
    # para cada individuo (pop_size),
    # para cada mediana del individuo (p),
    dist_clientes_a_inst = cost_matrix[:, poblacion]   # (n, pop_size, p)
    # Para cada cliente y  cada individuo, buscamos la distancia más pequeña entre todas sus medianas
    # Resultado: (n, pop_size)
    dist_min_por_cliente = np.min(dist_clientes_a_inst, axis=2)
    #Para cada individuo ind, sumamos dist_min_por_cliente[0, ind]  + dist_min_por_cliente[1, ind] + ... + dist_min_por_cliente[n-1, ind]
    # Resultado: (pop_size,)
    costos = np.sum(dist_min_por_cliente, axis=0)
    return costos.astype(float)

def poblacion_inicial(pop_size: int, n: int, p: int) -> np.ndarray:
    """
    Genera una población inicial aleatoria para el pMP.
    Args:
    pop_size : int
        Tamaño de la población (número de individuos).
    n : int
        Número total de instalaciones disponibles.
    p : int
        Número de medianas a seleccionar por individuo.

    Returns:
    poblacion : np.ndarray
        Matriz de tamaño (pop_size, p), donde cada fila contiene los índices
        de las instalaciones seleccionadas (0..n-1) para un individuo.
    """
    poblacion = np.zeros((pop_size, p), dtype=int)
    for i in range(pop_size):
        individuo = np.random.choice(n, size=p, replace=False)
        individuo.sort()
        poblacion[i, :] = individuo
    return poblacion

def poblacion_inicial_combinaciones(pop_size: int, facilities: int, p: int) -> np.ndarray:
    """
    Genera una población inicial de tamaño pop_size usando la clase Combinaciones,
    garantizando individuos únicos (combinaciones sin repetidos).

    Args:
    pop_size : int
        Número de individuos.
    facilities : int
        Número total de instalaciones posibles (0..n-1).
    p : int
        Número de instalaciones seleccionadas por individuo.

    Returns:
    poblacion : np.ndarray
        Matriz (pop_size, p) con cromosomas ordenados y sin duplicados.
    """
    combs = Combinaciones()
    combs.generar(n=facilities, m=p, tam=pop_size)   # llena combs.datos con 'pop_size' combinaciones

    # Convertimos el set de tuplas en un array (pop_size, p)
    poblacion = np.array(list(combs.datos), dtype=int)

    #si el orden del set no coincide con pop_size exacto
    if poblacion.shape[0] > pop_size:
        poblacion = poblacion[:pop_size, :]

    return poblacion

@njit(parallel=True, cache=True)
def selecciona_torneo(poblacion, fitness, num_elegidos, num_competidores=5, maximizar=False) -> np.ndarray:
    padres = np.empty((num_elegidos, poblacion.shape[1]), dtype=poblacion.dtype)
    num = int(num_competidores)
    for i in prange(num_elegidos):
        competidores = np.random.choice(np.arange(len(poblacion)), num)
        resultados = list([(0,0)]*num)
        for j in prange(num):
            k = competidores[j]
            resultados[j] = (fitness[k], k)
        resultados.sort(reverse= maximizar)
        padres[i] = poblacion[resultados[0][1]] # El que tuvo "mejor" calificación
    return padres

@njit(cache=True)
def ruleta(poblacion, fitness, num_elegidos, maximizar = False)-> np.ndarray: #Para poblaciones no muy grandes
    suma_fitness = sum(fitness)
    tam = len(fitness)
    # Calcula las probabilidades de selección para cada individuo
    probabilidades = [1 - f/suma_fitness for f in fitness]
    # Invierte las probabilidades en caso de querer minimizar el fitness
    if maximizar:
        probabilidades = [1 - p for p in probabilidades]
    
    seleccionados = np.empty((num_elegidos, poblacion.shape[1]))
    t = -1 #contador de elegidos
    for _ in range(num_elegidos):
        r = np.random.rand()
        # selección basada en la ruleta
        suma_prob = 0
        for i in range(tam):
            suma_prob += probabilidades[i]
            if r <= suma_prob:
                seleccionados[(t:=t+1)] = poblacion[i]
                break
    return seleccionados

def cruzamiento_intercambio(padre1: np.ndarray, padre2: np.ndarray, facilities : int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cruza dos padres intercambiando segmentos de sus genes no comunes.
    Algoritmo:
      1) Identifica genes comunes (fijos) en ambos padres.
      2) Construye vectores de intercambio con los genes no comunes.
      3) Elige aleatoriamente un punto de corte k.
      4) Intercambia segmentos de los vectores de intercambio para generar 2 hijos.
      IMPORTANTE : Si los padres son idénticos, genera hijos aleatorios.
    Args:
    padre1, padre2  : np.ndarray
        Cromosoma 1D de longitud p con índices únicos (0..n-1).
    facilities : int
        Número total de posibles instalaciones.
        
    Returns:
    hijo1, hijo2 : np.ndarray
        Cromosomas de longitud p .
    """
    # Tamaño de los padres
    p = padre1.size 

    # Genes comunes (fijos)
    v_fixed = np.intersect1d(padre1, padre2, assume_unique=True)
    
    # Vectores de intercambio (genes propios de cada padre)
    v_ex1 = np.setdiff1d(padre1, v_fixed, assume_unique=True)
    v_ex2 = np.setdiff1d(padre2, v_fixed, assume_unique=True)

    q = v_ex1.size

    # Caso trivial: si no hay parte intercambiable (padres idénticos),
    # los hijos seran copias aleatorias de instalaciones
    if q == 0:
        hijo1 = np.random.choice(facilities, size=p, replace=False)
        hijo2 = np.random.choice(facilities, size=p, replace=False)
        # Como trabajamos con conjuntos, es buena idea ordenar los genes
        hijo1 = np.sort(hijo1)
        hijo2 = np.sort(hijo2)
        
        return hijo1, hijo2 
    
    # Elegir punto de corte k entre 1 y q-1
    if q == 1:
        # Solo un gen en intercambio: intercambiarlo completo
        k = 1
    else:
        k = np.random.randint(1,q)  # valores entre 1 y q-1

    # Construir las partes intercambiadas
    # hijo1: comunes + ex1[0:k] + ex2[k:q]
    # hijo2: comunes + ex2[0:k] + ex1[k:q]
    parte1_h1 = v_ex1[:k]
    parte2_h1 = v_ex2[k:]
    parte1_h2 = v_ex2[:k]
    parte2_h2 = v_ex1[k:]

    hijo1 = np.concatenate([v_fixed, parte1_h1, parte2_h1])
    hijo2 = np.concatenate([v_fixed, parte1_h2, parte2_h2])
    
    # Ordenar genes para mantener consistencia
    hijo1 = np.sort(hijo1)
    hijo2 = np.sort(hijo2)

    return hijo1, hijo2

def mutacion_simple(individuo: np.ndarray, facilities: int) -> np.ndarray:
    """
    Reemplaza UNA instalación del individuo por otra que no esté presente.

    Args:
    individuo : np.ndarray
        Vector 1D de longitud p con índices únicos (0..n-1).
    facilities : int
        Número total de posibles instalaciones. 

    Returns:
    nuevo_individuo : np.ndarray
        Individuo mutado
    """
    # Copia para no modificar el original
    mutado = individuo.copy()
    p = mutado.size
    pos = np.random.randint(0,p)  # entero en [0, p-1]
    todas = np.arange(facilities, dtype=int)
    # Genes disponibles para insertar (los que no están actualmente)
    disponibles = np.setdiff1d(todas, mutado, assume_unique=True)
    # Reemplazar en la posición elegida
    mutado[pos] = np.random.choice(disponibles)
    
    return mutado

def mutation_local_search(individuo: np.ndarray, cost_matrix : np.ndarray, n : int) -> np.ndarray:
    """
    Mutación por búsqueda local:
    - Selecciona UN gen aleatorio del individuo.
    - Genera todos los vecinos reemplazando ese gen por instalaciones
      que NO están actualmente en el cromosoma.
    - Evalúa todos los vecinos de forma vectorizada.
    - Devuelve el mejor (menor costo) entre el individuo original y sus vecinos.

    Args:
    individuo : np.ndarray
        Cromosoma 1D (p,) con índices de instalaciones (0..n-1).
    n : int
        Número total de instalaciones posibles.
    cost_matrix : np.ndarray
        Matriz de distancias (n, n).

    Returns:
    mutado : np.ndarray
        Individuo mutado (o el original si ningún vecino mejora el fitness).
    """
    individuo = np.asarray(individuo, dtype=int).ravel()
    all_facilities = np.arange(n, dtype=int)
    p = individuo.size

    # Elegir una posición (gen) al azar
    pos = np.random.randint(0, p)

    # Conjunto H: instalaciones que NO están en el cromosoma
    mask = ~np.isin(all_facilities, individuo)
    H = all_facilities[mask]


    # Construir todos los vecinos de forma vectorizada
    # Cada vecino es igual al individuo, excepto en 'pos'
    num_vecinos = H.size
    vecinos = np.tile(individuo, (num_vecinos, 1))  # (num_vecinos, p)
    vecinos[:, pos] = H                             # reemplazar gen en 'pos'

    # Evaluar individuo original + todos los vecinos de una sola vez
    poblacion_ext = np.vstack([individuo, vecinos])     # (1 + num_vecinos, p)
    costos = evaluar_poblacion(poblacion_ext,cost_matrix)     # (1 + num_vecinos,)

    # Elegir el de menor costo
    idx_best = np.argmin(costos)

    if idx_best == 0:
        # Ningún vecino mejora al original
        return individuo.copy()
    else:
        # Devolver el mejor vecino
        return poblacion_ext[idx_best].copy()

def mutation_local_search_sample(individuo: np.ndarray,n: int,cost_matrix: np.ndarray, sample_frac: float ) -> np.ndarray:
    """
    Búsqueda local por muestreo sobre un solo gen del individuo.
    - Selecciona UNA posición aleatoria del cromosoma.
    - Construye el conjunto H de facilities que no están en el individuo.
    - Toma una muestra aleatoria de tamaño ≈ sample_frac * |H|.
    - Genera los vecinos cambiando solo esa posición por cada facility de la muestra.
    - Evalúa individuo original + vecinos muestreados.
    - Devuelve el de menor costo (si ninguno mejora, devuelve el original).

    Args:
    individuo : np.ndarray
        Cromosoma 1D (p,) con índices de instalaciones 0..n-1.
    n : int
        Número total de instalaciones.
    cost_matrix : np.ndarray
        Matriz de distancias. clienets -> facilities (n, n).
    sample_frac : float
        Proporción de facilities en H que se usarán para el muestreo (0 < sample_frac ≤ 1).

    Returns:
    mutado : np.ndarray
        Individuo mutado (o el original si no hubo mejora en la muestra).
    """
    individuo = np.asarray(individuo, dtype=int).ravel()
    p = individuo.size

    # Elegir la posición del gen a mutar
    pos = np.random.randint(0, p)

    # Facilities que NO están en el individuo
    all_facilities = np.arange(n, dtype=int)
    mask = ~np.isin(all_facilities, individuo)
    H = all_facilities[mask]
    m = H.size

    # Tamaño de la muestra (al menos 1 y como máximo m)
    sample_size = max(1, int(round(sample_frac * m)))
    sample_size = min(sample_size, m)

    # Seleccionar muestra aleatoria sin reemplazo
    muestra = np.random.choice(H, size=sample_size, replace=False)

    # Generar vecinos para la muestra (vectorizado)
    vecinos = np.tile(individuo, (sample_size, 1))  # (sample_size, p)
    vecinos[:, pos] = muestra

    # Evaluar original + vecinos muestreados
    poblacion_ext = np.vstack([individuo, vecinos])  # (1 + sample_size, p)
    costos = evaluar_poblacion(poblacion_ext, cost_matrix)  # (1 + sample_size,)

    # Elegir el mejor (menor costo)
    idx_best = int(np.argmin(costos))

    if idx_best == 0:
        # Ningún vecino de la muestra mejora al original
        return individuo.copy()
    else:
        # Devolvemos el mejor vecino de la muestra
        return poblacion_ext[idx_best].copy()

def criterio_parada_cv(frac_mejores: float,umbral_cv: float ,min_gener: int , maximizar: bool = False):
    """
    Devuelve una función criterio_parada  que:
    - Solo empieza a evaluar a partir de min_gener.
    - Cada 10 generaciones calcula el CV del 'frac_mejores' de la población.
    - Si el CV < umbral_cv -> True (parar).
    """
    def criterio_parada(generacion: int, fitness: np.ndarray) -> bool:
        # No parar muy pronto
        if generacion + 1 < min_gener:
            return False

        # Solo revisar cada 10 generaciones
        if (generacion + 1) % 10 != 0:
            return False

        f = np.asarray(fitness, dtype=float)
        pop_size = f.size
        k = max(1, int(frac_mejores * pop_size))

        # Ordenar fitness según maximizar|minimizar
        if maximizar:
            ordenados = np.sort(f)[::-1]   # descendente
        else:
            ordenados = np.sort(f)         # ascendente

        mejores = ordenados[:k]
        media = np.mean(mejores)
        if media == 0:
            return False  # evitar división por 0

        desvest = np.std(mejores)
        cv = desvest / abs(media)

        # print(f"[criterio_parada] gen {generacion+1}, CV={cv:.6e}")  # para debug 

        return cv < umbral_cv

    return criterio_parada




    
if __name__ == '__main__':
    #* PARÁMETROS DEL PROBLEMA *#
    MATRIX = np.load('datasets\pmed1.npy')
    CLIENTS = len(MATRIX)
    FACILITIES = len(MATRIX)
    P = 5# Número de medianas a seleccionar 
    TAM = 500 # Tamaño de la población

    criterio_parada = criterio_parada_cv(
    frac_mejores = 0.2,
    umbral_cv = 1e-4,
    min_gener = 50,
    maximizar=False
)

    # POB = poblacion_inicial(TAM*2, FACILITIES, P)
    POB = poblacion_inicial_combinaciones(TAM*2, FACILITIES, P)


    print(
        'El tiempo de ejecución fue: ',
        AlgoritmoGenetico(
            num_iteraciones= 500,
            tam= TAM,
            poblacion_inicial= POB,
            evaluacion= evaluar_poblacion,
            seleccion= ruleta,
            cruzamiento = cruzamiento_intercambio,
            mutacion=  mutacion_simple,
            mutacion_elite=  mutation_local_search_sample,
            prob_mutacion= 0.6,
            prob_cruzamiento= 0.4,
            para_evaluacion= {'cost_matrix': MATRIX},
            # para_seleccion= {'num_competidores': 12}, #torneo
            para_seleccion= {}, #ruleta
            para_cruzamiento= {'facilities': FACILITIES },
            para_mutacion= {'facilities': FACILITIES }, #mutación simple
            para_mutacion_elite= {'n': FACILITIES,'cost_matrix': MATRIX, 'sample_frac': 0.37  }, #mutación por búsqueda local
            criterio_parada= criterio_parada,
            maximizar= False
            ).run()['tiempo'],
            'minutos.')



