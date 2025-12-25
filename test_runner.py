import numpy as np
from Genetic_pMP import evaluar_poblacion, poblacion_inicial_combinaciones, selecciona_torneo, ruleta, cruzamiento_intercambio,mutacion_simple
import os
import pandas as pd
import openpyxl
import pickle
from funtion_pMP import genetic_algorithm_pMP


# Parámetros generales del experimento 
# Número de réplicas por instancia
REPLICAS = 10
# Parámetros del GA 
NUM_ITERACIONES = 600 # generaciones
POP_SIZE = 500   # tamaño de población
PROB_CRUZAMIENTO = 0.4
PROB_MUTACION = 0.6

# Parámetros de Parada (Estancamiento)
MIN_GENER=1090 # Corre al menos 100 gens
MAX_GEN_SIN_MEJORA=170  # Si no mejora en 170 gens  entonces el algoritmo se detiene

# Parámetros de Reinicio Inteligente
UMBRAL_CV_REINICIO=0.05
MAX_ESTANCAMIENTO_REINICIO=40
FUERZA_PERTURBACION=0.15
FRECUENCIA_CHEQUEO=20


MAXIMIZAR = False  # p-median => minimizar

# Carpeta donde se encuentran  las matrices .npy
DATASET_DIR = "datasets"
# Carpeta de salida para el Excel
OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Diccionario con las instancias de p-median y su valor de p
test = {
    "pmed1": 5,
    "pmed2": 10,
    "pmed3": 10,
    "pmed4": 20,
    "pmed5": 33,
    "pmed6": 5,
    "pmed7": 10,
    "pmed8": 20,
    "pmed9": 40,
    "pmed10": 67,
    "pmed11": 5,
    "pmed12": 10,
    "pmed13": 30,
    "pmed14": 60,
    "pmed15": 100,
    "pmed16": 5,
    "pmed17": 10,
    "pmed18": 40,
    "pmed19": 80,
    "pmed20": 133,
    "pmed21": 5,
    "pmed22": 10,
    "pmed23": 50,
    "pmed24": 100,
    "pmed25": 167,
    "pmed26": 5,
    "pmed27": 10,
    "pmed28": 60,
    "pmed29": 120,
    "pmed30": 200,
    "pmed31": 5,
    "pmed32": 10,
    "pmed33": 70,
    "pmed34": 140,
    "pmed35": 5,
    "pmed36": 10,
    "pmed37": 80,
    "pmed38": 5,
    "pmed39": 10,
    "pmed40": 90,
}

# Creación de un nuevo libro de Excel para almacenar los resultados de la instancia actual.
workbook = openpyxl.Workbook()
ws = workbook.active
ws.title = "Resultados_pmed"
# Encabezados de columna
ws.append(["test", "mejor_fitness", "mean_fitness", "median_fitness",  "std_fitness", "tiempo_promedio_min"])


 # Itera a través de cada test para evaluar el modelo
for inst_name, p in test.items():
    file_path = os.path.join(DATASET_DIR, f"{inst_name}.npy")

    if not os.path.exists(file_path):
        print(f"[AVISO] No se encontró el archivo {file_path}. Se salta la instancia.")
        continue

    print(f"\n=== Ejecutando {inst_name} (p={p}) ===")

    cost_matrix = np.load(file_path)

    fitness_replicas = []
    tiempos_replicas = []

    # Ejecuta el algoritmo genético múltiples veces (réplicas)
    for rep in range(REPLICAS):
         # Llamada al modelo de algoritmo genético con la configuración de hiperparámetros.
        tiempo, mejor_indiv, mejor_fit = genetic_algorithm_pMP(
            cost_matrix=cost_matrix,
            p=p,
            num_iteraciones=NUM_ITERACIONES,
            pop_size=POP_SIZE,
            seleccion= selecciona_torneo,
            cruzamiento=cruzamiento_intercambio,
            mutacion=mutacion_simple,
            para_seleccion={},     # se rellenan por defecto dentro de Hybrid_GA_pMP
            para_cruzamiento={},
            para_mutacion={},
            prob_cruzamiento=PROB_CRUZAMIENTO,
            prob_mutacion=PROB_MUTACION,
            max_estancamiento=MAX_ESTANCAMIENTO_REINICIO,
            max_gen_sin_mejora=MAX_GEN_SIN_MEJORA,
            umbral_cv=UMBRAL_CV_REINICIO,
            min_gener=MIN_GENER,
            frecuencia_chequeo=FRECUENCIA_CHEQUEO,
            fuerza=FUERZA_PERTURBACION,
            maximizar=MAXIMIZAR,
        )
        # Recopilación de resultados de cada réplica.
        fitness_replicas.append(mejor_fit)
        tiempos_replicas.append(tiempo)

    # Estadísticos para esta instancia
    fitness_replicas = np.array(fitness_replicas, dtype=float)
    tiempos_replicas = np.array(tiempos_replicas, dtype=float)
    # Cálculo de estadísticas sobre los resultados de todas las réplicas.
    mejor_fitness = float(np.min(fitness_replicas))    # mejor (mínimo) entre las  réplicas
    mean_fitness = float(np.mean(fitness_replicas))   # media de los resultados
    median_fitness = float(np.median(fitness_replicas)) # mediana de los resultados
    std_fitness = float(np.std(fitness_replicas))      # desviación estándar de los resultados
    tiempo_promedio = float(np.mean(tiempos_replicas)) # tiempo medio en minutos

    # Añade los resultados al archivo de Excel.
    ws.append([inst_name, mejor_fitness,mean_fitness, median_fitness,  std_fitness, tiempo_promedio])
    print(f"{inst_name}: best={mejor_fitness:.4f}, mean = {mean_fitness}, median ={median_fitness}, std={std_fitness}, "
          f"time_avg={tiempo_promedio} min")


#Guardar Excel
output_path = os.path.join(OUTPUT_DIR, "resultados_GA_pmed.xlsx")
workbook.save(output_path)
print(f"\nResultados guardados en: {output_path}")



