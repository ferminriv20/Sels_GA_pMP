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

def mutacion_con_inmigracion(individuo: np.ndarray, facilities: int) -> np.ndarray:
    """
    Combina mutación simple con inmigración aleatoria (como el algoritmo de la imagen).
    """
    # 5% de probabilidad de 'Cataclismo': Devolver un individuo totalmente nuevo
    if np.random.rand() < 0.05: 
        p = individuo.size
        # Generar individuo aleatorio nuevo
        nuevo = np.random.choice(facilities, size=p, replace=False)
        nuevo.sort()
        return nuevo

    # 95% de probabilidad: Mutación normal (cambiar 1 gen)
    # (Tu código de mutación simple aquí...)
    mutado = individuo.copy()
    p = mutado.size
    todas = np.arange(facilities)
    disponibles = np.setdiff1d(todas, mutado, assume_unique=True)
    pos = np.random.randint(0, p)
    mutado[pos] = np.random.choice(disponibles)
    mutado.sort()
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

def criterio_parada_estancamiento(max_gen_sin_mejora: int, min_gener: int):
    """
    Detiene el algoritmo SOLO si el mejor individuo no mejora 
    durante 'max_gen_sin_mejora' generaciones consecutivas.
    """
    estado = {'mejor_fit': float('inf'), 'contador': 0}

    def criterio(generacion, fitness):
        if generacion < min_gener:
            return False
            
        mejor_actual = np.min(fitness) # Asumiendo minimización
        
        # Si mejora (con pequeña tolerancia por errores de flotante)
        if mejor_actual < estado['mejor_fit'] - 1e-6:
            estado['mejor_fit'] = mejor_actual
            estado['contador'] = 0
            # print(f"  >> Mejora en gen {generacion}: {mejor_actual}")
        else:
            estado['contador'] += 1
            
        if estado['contador'] >= max_gen_sin_mejora:
            print(f"STOP: Estancamiento por {max_gen_sin_mejora} generaciones.")
            return True
        return False

    return criterio


def criterio_reinicio_inteligente(umbral_cv: float, max_estancamiento: int, frecuencia_chequeo: int):
    """
    Ordena reiniciar si:
    1. La población es demasiado homogénea (CV < umbral)
       O
    2. El mejor fitness no ha mejorado en 'max_estancamiento' generaciones.
    """
    # Estado mutable para rastrear el estancamiento
    estado = {
        'mejor_fit_historico': float('inf'),
        'contador_estancamiento': 0,
        'ultimo_reinicio': 0
    }

    def criterio(paso_ignorado, generacion, fitness):
        # Evitar chequeos en la misma generación del reinicio
        if generacion == estado['ultimo_reinicio']:
            return False

        # 1. ACTUALIZAR ESTADO DE ESTANCAMIENTO (Se hace en cada generación)
        mejor_actual = np.min(fitness)
        
        # Si hay mejora real (tolerancia 1e-6)
        if mejor_actual < estado['mejor_fit_historico'] - 1e-6:
            estado['mejor_fit_historico'] = mejor_actual
            estado['contador_estancamiento'] = 0
        else:
            estado['contador_estancamiento'] += 1

        # 2. VERIFICAR CONDICIONES (Solo cada X generaciones para ahorrar tiempo, 
        #    o inmediatamente si el estancamiento es crítico)
        
        # A) Criterio de Estancamiento (Prioridad Alta)
        if estado['contador_estancamiento'] >= max_estancamiento:
            print(f"   [ALERTA] Estancamiento por {max_estancamiento} gens. REINICIO FORZADO.")
            # Reseteamos contadores para dar tiempo a la nueva población
            estado['contador_estancamiento'] = 0 
            estado['ultimo_reinicio'] = generacion
            return True

        # B) Criterio de Diversidad (Solo en chequeos periódicos)
        if generacion > 0 and generacion % frecuencia_chequeo == 0:
            media = np.mean(fitness)
            if abs(media) > 1e-9:
                cv = np.std(fitness) / media
                # print(f"   [DEBUG] Gen {generacion}: CV={cv:.4f} | Estancamiento={estado['contador_estancamiento']}")
                
                if cv < umbral_cv:
                    print(f"   [ALERTA] Baja diversidad (CV={cv:.4f}). REINICIO SOLICITADO.")
                    estado['contador_estancamiento'] = 0 # Reset al reiniciar
                    estado['ultimo_reinicio'] = generacion
                    return True

        return False

    return criterio


def accion_reinicio(facilities: int, p: int):
    """
    Conserva al mejor individuo usando el fitness ya calculado (costo O(N)).
    Reinicia el resto.
    """
    #
    def accion(poblacion: np.ndarray, fitness: np.ndarray, ratio_ignorado: float):
        
        # 1. Encontrar al mejor (¡Sin evaluar! Solo buscando el min en el array existente)
        # Asumimos minimización. Si fuera maximización usar np.argmax
        idx_mejor = np.argmin(fitness)
        
        # Copiamos al líder para que no se pierda al sobrescribir
        gen_lider = poblacion[idx_mejor].copy()
        fit_lider = fitness[idx_mejor] # Solo para el print
        
        # 2. Generar nueva población (N-1 individuos)
        pop_size = len(poblacion)
        num_nuevos = pop_size - 1
        
        # Generar aleatorios
        genes_nuevos = poblacion_inicial_combinaciones(num_nuevos, facilities, p)
        
        # 3. Sobrescribir la población
        # Colocamos al líder en la posición 0
        poblacion[0] = gen_lider
        
        # Llenamos el resto con sangre nueva
        # (Usamos slicing seguro por si genes_nuevos trae menos elementos)
        limit = min(len(genes_nuevos), num_nuevos)
        poblacion[1 : 1+limit] = genes_nuevos[:limit]
        
        print(f"   >>> [REINICIO RÁPIDO] Líder (Fit: {fit_lider:.1f}) salvado. {limit} renovados.")

    return accion

def mutacion_geografica(individuo: np.ndarray, cost_matrix: np.ndarray, num_vecinos_cercanos: int = 100) -> np.ndarray:
    """
    Estrategia Inteligente:
    En lugar de cambiar una instalación por otra totalmente aleatoria (que puede estar
    al otro lado del mapa), la reemplaza por una instalación geográficamente CERCANA.
    
    Esto permite refinar la solución sin destruir la estructura de cobertura.
    """
    mutado = individuo.copy()
    n = cost_matrix.shape[0]
    p = mutado.size

    # 1. Seleccionar aleatoriamente una instalación a eliminar (Gen saliente)
    idx_eliminar = np.random.randint(0, p)
    facility_saliente = mutado[idx_eliminar]

    # 2. Buscar candidatos geográficos:
    # Miramos la fila de la matriz de costos correspondiente a la facility saliente.
    # Obtenemos los índices de las instalaciones más cercanas (menor costo/distancia).
    # Tomamos 'num_vecinos_cercanos' candidatos más próximos.
    # (Argsort devuelve los índices ordenados de menor a mayor distancia)
    vecinos_cercanos = np.argsort(cost_matrix[facility_saliente])[:num_vecinos_cercanos+p]

    # 3. Filtrar:
    # Solo nos sirven los que NO están ya en el individuo
    candidatos_validos = np.setdiff1d(vecinos_cercanos, mutado, assume_unique=True)

    if candidatos_validos.size == 0:
        # Fallback raro: si todos los cercanos ya están, usar aleatorio global
        todas = np.arange(n)
        candidatos_validos = np.setdiff1d(todas, mutado, assume_unique=True)

    # 4. Seleccionar uno de los candidatos cercanos
    nuevo_gen = np.random.choice(candidatos_validos)

    # 5. Intercambio
    mutado[idx_eliminar] = nuevo_gen
    mutado.sort() # Mantener orden
    
    return mutado


def accion_reinicio_porcentaje(facilities: int, p: int, porcentaje: float ):
    """
    Estrategia de Reinicio Parcial:
    1. Ordena la población por fitness.
    2. Conserva el 'porcentaje' (15%) de los mejores.
    3. Reemplaza el resto (85%) con individuos nuevos aleatorios.
    """
    def accion(poblacion: np.ndarray, fitness: np.ndarray, ratio_ignorado: float):
        pop_size = len(poblacion)
        
        # Calculamos cuántos sobreviven
        n_elite = int(pop_size * porcentaje)
        if n_elite < 1: n_elite = 1 # Seguridad: siempre guardar al menos al mejor
        
        n_nuevos = pop_size - n_elite
        
        # 1. Identificar a los mejores (índices ordenados de menor a mayor costo)
        indices_ordenados = np.argsort(fitness)
        indices_elite = indices_ordenados[:n_elite]
        
        # Copiamos sus genes para protegerlos de la sobreescritura
        elite_genes = poblacion[indices_elite].copy()
        mejor_fit_actual = fitness[indices_ordenados[0]]
        
        # 2. Generar sangre nueva para el resto
        # Usamos tu generador de combinaciones para evitar duplicados internos
        nuevos_genes = poblacion_inicial_combinaciones(n_nuevos, facilities, p)
        
        # 3. Aplicar el reinicio en la matriz original
        # A) Ponemos a la élite al principio (posiciones 0 a n_elite-1)
        poblacion[:n_elite] = elite_genes
        
        # B) Rellenamos el resto con los nuevos
        # Nota: Manejamos el caso donde 'combinaciones' devuelva menos individuos de los pedidos
        cantidad_real_nuevos = len(nuevos_genes)
        limite_llenado = min(cantidad_real_nuevos, n_nuevos)
        
        poblacion[n_elite : n_elite + limite_llenado] = nuevos_genes[:limite_llenado]
        
        print(f"   >>> [REINICIO] Se conservó el {porcentaje*100:.0f}% ({n_elite} indiv). Mejor: {mejor_fit_actual:.1f}. Renovados: {limite_llenado}")

    return accion

def accion_reinicio_perturbacion(facilities: int, p: int, fuerza: float = 0.3):
    """
    Reinicio por Hipermutación (Kick):
    1. Conserva al mejor individuo intacto.
    2. Al resto de la población la 'sacude' fuertemente:
       Cambia el 'fuerza' (ej. 30%) de las instalaciones de cada individuo por otras aleatorias.
    
    Ventaja: Mantiene la calidad estructural ganada pero rompe el estancamiento.
    """
    def accion(poblacion: np.ndarray, fitness: np.ndarray, ratio_ignorado: float):
        # 1. Identificar y salvar al mejor
        idx_mejor = np.argmin(fitness)
        elite = poblacion[idx_mejor].copy()
        fit_elite = fitness[idx_mejor]
        
        pop_size = len(poblacion)
        # Número de genes a cambiar en cada individuo
        n_cambios = int(p * fuerza)
        
        # Universo de todas las instalaciones
        all_facilities = np.arange(facilities)
        
        # 2. Perturbar a toda la población (menos al mejor temporalmente)
        for i in range(pop_size):
            # Trabajamos sobre el individuo in-place
            ind = poblacion[i]
            
            # a) Elegir posiciones al azar para eliminar
            idxs_borrar = np.random.choice(p, n_cambios, replace=False)
            
            # b) Buscar candidatos que NO estén en el individuo actual
            # (Usamos setdiff1d para exactitud)
            disponibles = np.setdiff1d(all_facilities, ind, assume_unique=True)
            
            # c) Elegir reemplazos
            # Si hay menos disponibles que cambios (raro), ajustamos
            n_real = min(len(disponibles), n_cambios)
            nuevos = np.random.choice(disponibles, n_real, replace=False)
            
            # d) Aplicar cambios
            ind[idxs_borrar[:n_real]] = nuevos
            ind.sort() # Siempre mantener ordenado
            
        # 3. Restaurar al mejor individuo en la posición 0 para no perderlo
        poblacion[0] = elite
        
        print(f"   >>> [REINICIO PERTURBACIÓN] Líder ({fit_elite:.1f}) salvado. Población sacudida un {fuerza*100:.0f}%.")

    return accion

##Adicional

def generar_semilla_greedy_rapida(n: int, p: int, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Versión optimizada.
    Costo computacional: O(p * n). Muy rápido.
    """
    solucion = []
    # Mascara de candidatos disponibles
    candidatos_mask = np.ones(n, dtype=bool) 
    
    # Vector de distancias mínimas actuales (inicia en infinito)
    # Shape: (n_clientes,)
    current_min_dists = np.full(n, np.inf)

    for _ in range(p):
        # Indices de los candidatos disponibles
        indices_candidatos = np.where(candidatos_mask)[0]
        
        # --- VECTORIZACIÓN INTELLIGENTE ---
        # 1. Tomar las filas de la matriz de costos correspondientes a los candidatos
        # Shape: (n_candidatos, n_clientes)
        costos_candidatos = cost_matrix[indices_candidatos]
        
        # 2. Calcular hipotéticamente cómo quedarían las distancias si agregamos cada candidato
        # Comparamos la distancia actual vs la nueva posible.
        # Broadcasting: (1, n_clientes) vs (n_candidatos, n_clientes)
        nuevas_distancias = np.minimum(current_min_dists[None, :], costos_candidatos)
        
        # 3. Sumar costos para ver cuál es el mejor
        costos_totales = np.sum(nuevas_distancias, axis=1)
        
        # 4. Elegir el mejor
        idx_relativo_mejor = np.argmin(costos_totales)
        mejor_candidato = indices_candidatos[idx_relativo_mejor]
        
        # 5. Actualizar estado
        solucion.append(mejor_candidato)
        candidatos_mask[mejor_candidato] = False
        current_min_dists = nuevas_distancias[idx_relativo_mejor] # Guardamos las nuevas distancias base

    return np.array(sorted(solucion))

def generar_semilla_greedy_aleatorizada(n: int, p: int, cost_matrix: np.ndarray, alpha: int = 5) -> np.ndarray:
    """
    Greedy Aleatorizado (GRASP construction):
    En lugar de tomar siempre el mejor candidato, elige uno al azar 
    de los 'alpha' mejores candidatos en cada paso.
    Esto genera diversidad estructural manteniendo alta calidad.
    """
    solucion = []
    current_min_dists = np.full(n, np.inf)
    candidatos_mask = np.ones(n, dtype=bool)

    for _ in range(p):
        indices_candidatos = np.where(candidatos_mask)[0]
        
        # Evaluación vectorizada (igual que tu greedy rápido)
        costos_candidatos = cost_matrix[indices_candidatos]
        nuevas_distancias = np.minimum(current_min_dists[None, :], costos_candidatos)
        costos_totales = np.sum(nuevas_distancias, axis=1)
        
        # --- LA DIFERENCIA ESTÁ AQUÍ ---
        # No tomamos el argmin (el mejor absoluto), sino los 'alpha' mejores
        # Si hay menos candidatos que alpha, tomamos todos
        limit = min(alpha, len(costos_totales))
        
        # Obtenemos índices de los mejores (partition es más rápido que sort total)
        # argpartition pone los k menores al principio
        indices_mejores_locales = np.argpartition(costos_totales, limit-1)[:limit]
        
        # Elegimos uno al azar de esos mejores
        eleccion_idx = np.random.choice(indices_mejores_locales)
        
        mejor_candidato = indices_candidatos[eleccion_idx]
        
        solucion.append(mejor_candidato)
        candidatos_mask[mejor_candidato] = False
        current_min_dists = nuevas_distancias[eleccion_idx]

    return np.array(sorted(solucion))


def pulido_final_intensivo(individuo: np.ndarray, cost_matrix: np.ndarray, num_vecinos: int = 300):
    """
    Aplica una búsqueda local intensiva al final del algoritmo.
    Intenta intercambiar cada gen del individuo con sus 'num_vecinos' más cercanos.
    Se repite hasta que no haya mejoras.
    """
    mejor_ind = individuo.copy()
    n = cost_matrix.shape[0]
    p = mejor_ind.size
    
    mejor_costo = evaluar_poblacion(np.array([mejor_ind]), cost_matrix)[0]
    print(f" >> Iniciando Pulido Final. Costo inicial: {mejor_costo}")

    mejora = True
    iteracion = 0
    
    while mejora:
        mejora = False
        iteracion += 1
        
        # Iteramos sobre cada instalación que tiene el individuo
        for i in range(p):
            facility_actual = mejor_ind[i]
            
            # Buscamos candidatos cercanos a esta facility que NO estén en la solución
            candidatos_raw = np.argsort(cost_matrix[facility_actual])
            candidatos = []
            count = 0
            for cand in candidatos_raw:
                if cand not in mejor_ind:
                    candidatos.append(cand)
                    count += 1
                if count >= num_vecinos:
                    break
            
            # Probamos intercambiar facility_actual por cada candidato
            for cand in candidatos:
                vecino = mejor_ind.copy()
                vecino[i] = cand
                # Nota: No ordenamos 'vecino' aquí para ahorrar tiempo, 
                # pero para evaluar_poblacion no importa el orden si es por índices.
                
                costo_vecino = evaluar_poblacion(np.array([vecino]), cost_matrix)[0]
                
                if costo_vecino < mejor_costo - 1e-6:
                    print(f"    [Pulido] Mejora encontrada: {mejor_costo} -> {costo_vecino}")
                    mejor_costo = costo_vecino
                    mejor_ind = vecino
                    mejor_ind.sort() # Mantenemos ordenado
                    mejora = True
                    break # Restart loop on improvement (First Improvement strategy)
            
            if mejora: break
            
    return mejor_ind, mejor_costo

def pulido_final_rapido(individuo: np.ndarray, cost_matrix: np.ndarray, num_vecinos: int = 20):
    """
    Búsqueda local restringida geográficamente.
    Solo revisa los 'num_vecinos' más cercanos a cada instalación.
    """
    mejor_ind = individuo.copy()
    p = mejor_ind.size
    
    # Evaluar estado inicial
    mejor_costo = evaluar_poblacion(np.array([mejor_ind]), cost_matrix)[0]
    
    mejora = True
    while mejora:
        mejora = False
        # Para cada instalación que tenemos abierta...
        for i in range(p):
            facility_actual = mejor_ind[i]
            
            # 1. TRUCO DE VELOCIDAD:
            # Solo consideramos candidatos que sean "vecinos cercanos" de la instalación actual.
            # (Ya están pre-ordenados en cost_matrix si usamos argsort, o lo hacemos al vuelo)
            # Tomamos los K más cercanos.
            vecinos_cercanos = np.argsort(cost_matrix[facility_actual])
            
            candidatos_revisados = 0
            for cand in vecinos_cercanos:
                if cand in mejor_ind: continue # Si ya está, saltar
                
                # Crear vecino hipotético
                vecino_propuesto = mejor_ind.copy()
                vecino_propuesto[i] = cand
                
                # Evaluación Delta (Rápida) o Completa
                # Aquí usamos la completa porque son pocas evaluaciones
                nuevo_costo = evaluar_poblacion(np.array([vecino_propuesto]), cost_matrix)[0]
                
                if nuevo_costo < mejor_costo - 1e-6:
                    mejor_costo = nuevo_costo
                    mejor_ind = vecino_propuesto
                    mejor_ind.sort()
                    mejora = True
                    # Estrategia First Improvement: si mejoramos, reiniciamos el bucle principal
                    break 
                
                candidatos_revisados += 1
                if candidatos_revisados >= num_vecinos:
                    break # Ya miramos los 20 más cercanos, pasamos a la siguiente facility
            
            if mejora: break

    return mejor_ind, mejor_costo

def pulido_VND(individuo: np.ndarray, cost_matrix: np.ndarray):
    """
    Mejora: Variable Neighborhood Descent.
    Alterna entre vecindario k=1 (rápido) y k=2 (lento pero desatasca).
    """
    mejor_ind = individuo.copy()
    n = cost_matrix.shape[0]
    p = mejor_ind.size
    mejor_costo = evaluar_poblacion(np.array([mejor_ind]), cost_matrix)[0]
    
    print(f" >> Iniciando VND (Costo inicial: {mejor_costo})...")
    
    k = 1
    max_k = 2 # Niveles de vecindad
    
    while k <= max_k:
        mejora = False
        
        if k == 1:
            # --- SWAP 1 (Tu pulido rápido optimizado) ---
            # Revisa vecinos cercanos
            for i in range(p):
                fac_actual = mejor_ind[i]
                # Solo revisar los 20 más cercanos para velocidad
                candidatos = np.argsort(cost_matrix[fac_actual])[:20] 
                
                for cand in candidatos:
                    if cand in mejor_ind: continue
                    
                    vecino = mejor_ind.copy()
                    vecino[i] = cand
                    costo_vecino = evaluar_poblacion(np.array([vecino]), cost_matrix)[0]
                    
                    if costo_vecino < mejor_costo - 1e-6:
                        mejor_costo = costo_vecino
                        mejor_ind = vecino
                        mejor_ind.sort()
                        mejora = True
                        print(f"   [VND k=1] Mejora: {mejor_costo}")
                        break
                if mejora: break
        
        elif k == 2:
            # --- SWAP 2 (Restringido) ---
            # Intenta intercambiar 2 instalaciones a la vez para salir de óptimo local
            # Solo hacemos unos intentos aleatorios inteligentes para no tardar años
            print("   [VND] Intentando movimiento doble (k=2)...")
            num_intentos = 50 
            for _ in range(num_intentos):
                # Elegir 2 al azar para sacar
                idxs = np.random.choice(p, 2, replace=False)
                
                # Elegir 2 cercanos para entrar
                cand1 = np.argsort(cost_matrix[mejor_ind[idxs[0]]])[1] # El vecino más cercano
                cand2 = np.argsort(cost_matrix[mejor_ind[idxs[1]]])[1]
                
                if cand1 not in mejor_ind and cand2 not in mejor_ind and cand1 != cand2:
                    vecino = mejor_ind.copy()
                    vecino[idxs[0]] = cand1
                    vecino[idxs[1]] = cand2
                    costo_vecino = evaluar_poblacion(np.array([vecino]), cost_matrix)[0]
                    
                    if costo_vecino < mejor_costo - 1e-6:
                        mejor_costo = costo_vecino
                        mejor_ind = vecino
                        mejor_ind.sort()
                        mejora = True
                        print(f"   [VND k=2] ¡DESBLOQUEO! Mejora: {mejor_costo}")
                        break

        if mejora:
            k = 1 # Si mejoramos, volvemos al vecindario rápido
        else:
            k += 1 # Si no, profundizamos

    return mejor_ind, mejor_costo

if __name__ == '__main__':
    #* PARÁMETROS DEL PROBLEMA *#
    MATRIX = np.load('datasets\pmed18.npy')
    CLIENTS = len(MATRIX)
    FACILITIES = len(MATRIX)
    P = 40# Número de medianas a seleccionar 
    TAM = 600 # Tamaño de la población

    criterio_parada_config= criterio_parada_estancamiento(max_gen_sin_mejora=170, min_gener=100)
    criterio_reinicio_config = criterio_reinicio_inteligente(umbral_cv=0.05, max_estancamiento=40, frecuencia_chequeo=20)
    accion_reinicio_config = accion_reinicio_perturbacion(facilities= FACILITIES, p=P, fuerza=0.15)

    # POB = poblacion_inicial(TAM*2, FACILITIES, P)
    # POB = poblacion_inicial_combinaciones(TAM*2, FACILITIES, P)
    # 1. GENERAR POBLACIÓN HÍBRIDA
    print("Generando población inicial...")
    # A) El 99% es aleatorio (usando tu método de combinaciones o random)
    POB = poblacion_inicial_combinaciones(TAM, FACILITIES, P)
    # Inyectar el 10% de la población con semillas GRASP
    num_semillas = int(TAM * 0.10) 
    print(f" >> Inyectando {num_semillas} semillas GRASP variadas...")
    
    for i in range(num_semillas):
        # alpha=5 significa que elegimos entre los 5 mejores en cada paso
        # Variamos alpha ligeramente para más caos
        POB[i] = generar_semilla_greedy_aleatorizada(FACILITIES, P, MATRIX, alpha=np.random.randint(3, 8))
    # B) Inyectamos la Semilla Greedy (El "Súper Individuo")
    #Esto le da al GA una pista muy fuerte de dónde buscar.
    # semilla = generar_semilla_greedy_rapida(FACILITIES, P, MATRIX)
    # POB[0] = semilla # Reemplazamos al primero con la semilla

    # print(
    #     'El tiempo de ejecución fue: ',
        
    print("Ejecutando GA...")
    resultado_ga = AlgoritmoGenetico(
            num_iteraciones= 600,
            tam= TAM,
            poblacion_inicial= POB,
            evaluacion= evaluar_poblacion,
            seleccion= selecciona_torneo,
            cruzamiento = cruzamiento_intercambio,
            mutacion=  mutacion_geografica,
            mutacion_elite=  mutation_local_search,
            prob_mutacion= 0.4,
            prob_cruzamiento= 0.6,
            para_evaluacion= {'cost_matrix': MATRIX},
            para_seleccion= {'num_competidores': 8}, #torneo
            # para_seleccion= {}, #ruleta
            para_cruzamiento= {'facilities': FACILITIES },
            # para_mutacion= {'facilities': FACILITIES }, #mutación simple              
            para_mutacion= {'cost_matrix': MATRIX, 'num_vecinos_cercanos': 20 }, #mutación geográfica
            para_mutacion_elite= {'cost_matrix': MATRIX,'n': FACILITIES }, #mutación por búsqueda local
            criterio_parada= criterio_parada_config,
            criterio_reinicio= criterio_reinicio_config,
            reinicio_poblacion= accion_reinicio_config,
            ratio_reinicio= 0.0,
            paso_reinicio= 0.0,
            maximizar= False
            ).run()
    mejor_individuo_ga = resultado_ga['mejor']
    mejor_fitness_ga = resultado_ga['resultado']
    tiempo_ga = resultado_ga['tiempo']

    # 3. PULIDO FINAL (POST-OPTIMIZACIÓN)
    # Aquí es donde cierras el GAP final
    mejor_individuo_final, mejor_fitness_final = pulido_VND(mejor_individuo_ga, MATRIX)

    print("\n" + "="*50)
    print(f"RESULTADO FINAL:")
    print(f"GA Fitness: {mejor_fitness_ga}")
    print(f"Fitness tras Pulido: {mejor_fitness_final}")
    print(f"Tiempo GA: {tiempo_ga} min")
    print("="*50)



