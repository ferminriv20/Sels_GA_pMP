import numpy as np
from GeneticAlgorithm_V2 import AlgoritmoGenetico
from Genetic_pMP import evaluar_poblacion, poblacion_inicial_combinaciones, selecciona_torneo, ruleta, cruzamiento_intercambio, mutation_local_search,mutacion_simple, criterio_parada_estancamiento, criterio_reinicio_inteligente,  accion_reinicio_perturbacion, generar_semilla_greedy_aleatorizada

def genetic_algorithm_pMP( cost_matrix: np.ndarray, p: int, num_iteraciones: int , pop_size: int , seleccion , cruzamiento , mutacion ,   para_seleccion :dict , para_cruzamiento: dict, para_mutacion : dict,  prob_cruzamiento: float , prob_mutacion: float , max_estancamiento: int ,max_gen_sin_mejora: int ,umbral_cv: float ,min_gener: int , frecuencia_chequeo: int,  fuerza: int,  maximizar: bool = False) -> dict:
    """
    Funcion que encapsula el algoritmo genético  para el  pMP.
    """

    # número total de instalaciones
    FACILITIES = cost_matrix.shape[0]
    # Población inicial usando combinaciones únicas
    POB = poblacion_inicial_combinaciones(pop_size, FACILITIES, p)
    
    num_semillas = int(pop_size* 0.10) 
    print(f" >> Inyectando {num_semillas} semillas GRASP variadas...")
    
    for i in range(num_semillas):
        # alpha=5 significa que elegimos entre los 5 mejores en cada paso
        # Variamos alpha ligeramente para más caos
        POB[i] = generar_semilla_greedy_aleatorizada(FACILITIES, p, cost_matrix, alpha=np.random.randint(3, 8))

    # Selección: aceptar string o función y construir parámetros por defecto si es necesario
    para_seleccion = para_seleccion or {}
    if isinstance(seleccion, str):
        if seleccion == 'ruleta':
            seleccion_fun = ruleta
        elif seleccion  == 'selecciona_torneo':
            seleccion_fun = selecciona_torneo
            # poner valor por defecto si no se pasó
            para_seleccion = {**{"num_competidores": 8}, **para_seleccion}
        else:
            raise ValueError("param 'seleccion' debe ser 'ruleta' o 'selecciona_torneo' o una función callable")
    elif callable(seleccion):
        seleccion_fun = seleccion
        if seleccion is selecciona_torneo and not para_seleccion:
            para_seleccion = {"num_competidores": 12}
    else:
        raise ValueError("param 'seleccion' inválido")

    # Cruzamiento: aceptar string o función
    para_cruzamiento = para_cruzamiento or {}
    if isinstance(cruzamiento, str):
        if cruzamiento == 'cruzamiento_intercambio':
            cruz_fun = cruzamiento_intercambio
            para_cruzamiento = {**{"facilities": FACILITIES}, **para_cruzamiento}
        else:
            raise ValueError("param 'cruzamiento' debe ser 'cruzamiento_intercambio' o una función callable")
    elif callable(cruzamiento):
        cruz_fun = cruzamiento
        para_cruzamiento = para_cruzamiento or {}
        if cruz_fun is cruzamiento_intercambio and "facilities" not in para_cruzamiento:
            para_cruzamiento["facilities"] = FACILITIES
    else:
        raise ValueError("param 'cruzamiento' inválido")

    # Mutación: aceptar string o función
    para_mutacion = para_mutacion or {}
    if isinstance(mutacion, str):
        if mutacion == "mutation_local_search_":
            mut_fun = mutation_local_search
            para_mutacion = {**{"n": FACILITIES, "cost_matrix": cost_matrix}, **para_mutacion}
        elif mutacion == "mutacion_simple":
            mut_fun = mutacion_simple
            para_mutacion = {**{"facilities": FACILITIES}, **para_mutacion}
        else:
            raise ValueError("param 'mutacion' debe ser 'mutation_local_search_sample', 'mutacion_simple' o una función callable")
    elif callable(mutacion):
        mut_fun = mutacion
        # si es la función de sample y no hay parámetros, añadir por defecto
        if mutacion is mutation_local_search and not para_mutacion:
            para_mutacion = {"n": FACILITIES, "cost_matrix": cost_matrix}
        elif mutacion is mutacion_simple and not para_mutacion:
            para_mutacion = {"facilities": FACILITIES}
        else:
            para_mutacion = para_mutacion or {}
    else:
        raise ValueError("param 'mutacion' inválido")
    
     # MUTACIÓN ELITE: siempre la búsqueda local por muestra sobre el mejor
    mutacion_elite_fun = mutation_local_search
    para_mutacion_elite = {"n": FACILITIES, "cost_matrix": cost_matrix}
    # Criterio de parada
    
    criterio_parada= criterio_parada_estancamiento(max_gen_sin_mejora, min_gener)
    criterio_reinicio = criterio_reinicio_inteligente(umbral_cv, max_estancamiento, frecuencia_chequeo)
    accion_reinicio = accion_reinicio_perturbacion(FACILITIES, p, fuerza)


    # Construir y ejecutar el algoritmo genético
    result = AlgoritmoGenetico(
        num_iteraciones=num_iteraciones,
        tam=pop_size,
        poblacion_inicial= POB,
        evaluacion=evaluar_poblacion,
        seleccion= seleccion_fun,
        cruzamiento= cruz_fun,
        mutacion= mut_fun,
        prob_mutacion=prob_mutacion,
        prob_cruzamiento=prob_cruzamiento,
        para_evaluacion={"cost_matrix": cost_matrix},
        para_seleccion=para_seleccion,
        para_cruzamiento=para_cruzamiento,
        para_mutacion=para_mutacion,
        maximizar=maximizar,
        mutacion_elite=mutacion_elite_fun,
        para_mutacion_elite=para_mutacion_elite,
        criterio_parada= criterio_parada,
        criterio_reinicio= criterio_reinicio,
        reinicio_poblacion= accion_reinicio,
        ratio_reinicio= 0.0,
        paso_reinicio= 0.0,
    ).run()


    return result["tiempo"] , result["mejor"] ,result["resultado"] 


#prueba

if __name__ == "__main__":
    print("--- Prueba Hybrid_GA_pMP ---")

    p = 40
    TAM =500
    MATRIX = np.load(r'datasets\pmed18.npy')
    

    print(genetic_algorithm_pMP(
        cost_matrix=MATRIX,
        p=p,
        num_iteraciones=500,
        pop_size=TAM,
        seleccion= selecciona_torneo,
        cruzamiento=cruzamiento_intercambio,
        mutacion=mutacion_simple ,
        para_seleccion={},
        para_cruzamiento={},
        para_mutacion={},
        prob_cruzamiento=0.4,
        prob_mutacion=0.6,
        max_estancamiento=40,
        max_gen_sin_mejora=170,
        umbral_cv=0.05,
        min_gener=90,
        frecuencia_chequeo=20,
        fuerza=0.15,
        maximizar=False,
    )[2])

 


