import numpy as np
import time

class AlgoritmoGenetico:
    def __init__(
            self, 
            num_iteraciones: int, 
            tam: int, 
            poblacion_inicial: np.ndarray,
            evaluacion: callable,
            seleccion: callable,
            cruzamiento: callable,
            mutacion: callable,
            prob_mutacion: float, 
            prob_cruzamiento: float, 
            para_evaluacion= dict(),
            para_seleccion= dict(),
            para_cruzamiento= dict(),
            para_mutacion= dict(),
            cruzar_pares: bool = False, 
            criterio_parada: callable= lambda x,y: False, # (generacion, fitness)
            criterio_reinicio: callable = lambda x,y,z: False, # (paso_reinicio, generacion, fitness)
            reinicio_poblacion: callable = False, # (poblacion, ratio)
            ratio_reinicio: float = 0.95,
            paso_reinicio: float = 0.2, # Cada tanto porcentaje del número de iteracinoes se permite hacer reinicio
            mejor: np.ndarray = None,
            maximizar: bool = False,
            mutacion_elite: callable = None,
            para_mutacion_elite: dict = None,
            ):
        
        self.num_iteraciones = num_iteraciones
        self.tam = tam
        self.poblacion = poblacion_inicial
        self.criterio_parada = criterio_parada
        self.metodo_evaluacion = evaluacion
        self.metodo_seleccion = seleccion
        self.metodo_cruzamiento = cruzamiento
        self.metodo_mutacion = mutacion
        self.prob_mutacion = prob_mutacion
        self.prob_cruzamiento = prob_cruzamiento
        self.para_evaluacion = para_evaluacion
        self.para_seleccion = para_seleccion
        self.para_cruzamiento = para_cruzamiento
        self.para_mutacion = para_mutacion
        self.cruzar_pares = cruzar_pares
        self.criterio_reinicio = criterio_reinicio
        self.reinicio_poblacion = reinicio_poblacion
        self.ratio_reinicio = ratio_reinicio
        self.paso_reinicio = paso_reinicio
        self.mejor = mejor
        self.maximizar = maximizar
        self.metodo_mutacion_elite = mutacion_elite
        self.para_mutacion_elite = para_mutacion_elite if para_mutacion_elite is not None else {}

    def evaluacion(self, pob):
        return self.metodo_evaluacion(pob, **self.para_evaluacion)

    def seleccion(self, pob, fitness):
        return self.metodo_seleccion(pob, fitness, self.tam, maximizar= self.maximizar, **self.para_seleccion)

    def mutacion(self, individuo):
        return self.metodo_mutacion(individuo, **self.para_mutacion)

    def cruzamiento(self, padre1, padre2):
        return self.metodo_cruzamiento(padre1, padre2, **self.para_cruzamiento)
    
    def cruzamiento_total(self, seleccionados):
        return self.metodo_cruzamiento(seleccionados, **self.para_cruzamiento)
    
    def mutacion_elite(self, individuo):
        """
        Aplica la mutación especial para el mejor individuo.
        Si no se definió mutacion_elite, usa la mutación normal.
        """
        if self.metodo_mutacion_elite is None:
            return self.metodo_mutacion(individuo, **self.para_mutacion)
        return self.metodo_mutacion_elite(individuo, **self.para_mutacion_elite)


    def run(self):
        start = time.time()
        print('Iniciando Algoritmo... \n\n')
        
        # Inicio de Iteraciones 
        for generacion in range(self.num_iteraciones):
            # Evalúa la aptitud de cada individuo en la población
            fitness = self.evaluacion(self.poblacion)

            # Verifica el criterio de reinicio poblacional
            if self.criterio_reinicio(self.paso_reinicio, generacion, fitness):
                self.reinicio_poblacion(self.poblacion, self.ratio_reinicio)
                fitness = self.evaluacion(self.poblacion)
                print(f' ### Reinicio de Población en la generación {generacion}\n')

            # Se busca el mejor individuo de la población actual
            #mejor = self.poblacion[np.argmax(fitness)] if self.maximizar else self.poblacion[np.argmin(fitness)].copy()
            #self.mejor[:] = mejor
            
            if self.maximizar:
                idx_mejor = np.argmax(fitness)
            else:
                idx_mejor = np.argmin(fitness)

            mejor = self.poblacion[idx_mejor].copy()
            
            # Aplicar mutación al mejor individuo cada 10 generaciones
            if (generacion + 1) % 10 == 0:
                mejor_mutado = self.mutacion_elite(mejor)
                mejor = mejor_mutado.copy()
                self.poblacion[idx_mejor] = mejor


            # Verifica el criterio de parada
            if self.criterio_parada(generacion, fitness):
                break
            
            
            
            # Realiza la selección de individuos para la reproducción
            seleccionados = self.seleccion(self.poblacion, fitness)

            # Realiza el cruzamiento para generar descendencia
            descendencia = []
            if self.cruzar_pares:
                descendencia = self.cruzamiento_total(seleccionados)
            else:
                for _ in range(self.tam + 1):
                    if np.random.rand() > self.prob_cruzamiento:
                        continue # No se realiza el cruzamiento si la probabilidad es menor a la configurada.
                    padre1, padre2 = np.random.choice(self.tam, size=2, replace=False) # índices de los padres
                    hijos = np.array(self.cruzamiento(seleccionados[padre1], seleccionados[padre2]))

                    if hijos.ndim == 1: # Validará si del cruzamiento se genera un solo individuo o un arreglo de individuos.
                        descendencia.append(hijos)
                    else:
                        descendencia.extend(hijos)

            # Realiza la mutación en la descendencia
            for i, individuo in enumerate(descendencia):
                if np.random.rand() < self.prob_mutacion:
                    descendencia[i] = self.mutacion(individuo)
        
            # Seleccionar individuos para ser reemplazados por la descendencia
            for i in range(len(descendencia)):
                indice_reemplazo = np.random.randint(len(self.poblacion))
                self.poblacion[indice_reemplazo] = descendencia[i]
      
            idx_peor = np.argmax(fitness) if not self.maximizar else np.argmin(fitness)
            # Se añade el mejor individuo de la descendencia a la población actual
            self.poblacion[idx_peor] = mejor

            # Se reporta el avance del algoritmo
            if (generacion+1) % 10 == 0: 
                print(f' Reporte Hasta Iteración #{generacion+1} ...')
                fitness = self.evaluacion(self.poblacion)
                self.reporte_de_estado(fitness, mejor= self.poblacion[np.argmax(fitness)] if self.maximizar else self.poblacion[np.argmin(fitness)])
            
        # Devuelve el mejor individuo encontrado
        mejor_individuo = self.poblacion[np.argmax(fitness)] if self.maximizar else self.poblacion[np.argmin(fitness)]
        resultado = np.amax(fitness) if self.maximizar else np.amin(fitness)

        print('\n', f'Se realizaron un total de {generacion+1} iteraciones y se encontró que el individuo más apto fue:')
        print(f'  -> Individuo: \n{mejor_individuo}, \ncon una aptitud de {np.amax(fitness) if self.maximizar else np.amin(fitness)}')
        print(' FIN EJECUCIÓN '.center(100, ':'))
        end = time.time()
        return {'pob': self.poblacion, 'aptitudes': fitness, 'mejor': mejor_individuo, 'resultado': resultado, 'tiempo': round((end-start)/60, 2)}
    
    # Método para reportar el estado del algoritmo:
    def reporte_de_estado(self, fitness, mejor):
        print(f'Mejor individuo: {mejor}\n')
        print(f'Aptitud: {np.amax(fitness) if self.maximizar else np.amin(fitness)}\n')
    #   print(f'Tamaño de la población: {len(fitness)}')
        print('-'*80)
        return None


    def __str__(self):
        return self.poblacion
    # Se crean los métodos de acceso a los atributos de la clase:
    @property
    def num_iteraciones(self):
        return self._num_iteraciones
    
    @num_iteraciones.setter
    def num_iteraciones(self, num_iteraciones):
        self._num_iteraciones = num_iteraciones

    @property
    def tam(self):
        return self._tam
    
    @tam.setter
    def tam(self, tam):
        self._tam = tam

    @property
    def poblacion(self):
        return self._poblacion
    
    @poblacion.setter
    def poblacion(self, poblacion):
        self._poblacion = poblacion

    @property
    def criterio_parada(self):
        return self._criterio_parada
    
    @criterio_parada.setter
    def criterio_parada(self, criterio_parada):
        self._criterio_parada = criterio_parada

    @property
    def metodo_evaluacion(self):
        return self._metodo_evaluacion
    
    @metodo_evaluacion.setter
    def metodo_evaluacion(self, metodo_evaluacion):
        self._metodo_evaluacion = metodo_evaluacion

    @property
    def metodo_seleccion(self):
        return self._metodo_seleccion
    
    @metodo_seleccion.setter
    def metodo_seleccion(self, metodo_seleccion):
        self._metodo_seleccion = metodo_seleccion

    @property
    def metodo_cruzamiento(self):
        return self._metodo_cruzamiento
    
    @metodo_cruzamiento.setter
    def metodo_cruzamiento(self, metodo_cruzamiento):
        self._metodo_cruzamiento = metodo_cruzamiento

    @property
    def metodo_mutacion(self):
        return self._metodo_mutacion
    
    @metodo_mutacion.setter
    def metodo_mutacion(self, metodo_mutacion):
        self._metodo_mutacion = metodo_mutacion

    @property
    def prob_mutacion(self):
        return self._prob_mutacion
    
    @prob_mutacion.setter
    def prob_mutacion(self, prob_mutacion):
        self._prob_mutacion = prob_mutacion

    @property
    def prob_cruzamiento(self):
        return self._prob_cruzamiento
    
    @prob_cruzamiento.setter
    def prob_cruzamiento(self, prob_cruzamiento):
        self._prob_cruzamiento = prob_cruzamiento

    @property
    def para_evaluacion(self):
        return self._para_evaluacion
    
    @para_evaluacion.setter
    def para_evaluacion(self, para_evaluacion):
        self._para_evaluacion = para_evaluacion

    @property
    def para_seleccion(self):
        return self._para_seleccion
    
    @para_seleccion.setter
    def para_seleccion(self, para_seleccion):
        self._para_seleccion = para_seleccion

    @property
    def para_cruzamiento(self):
        return self._para_cruzamiento
    
    @para_cruzamiento.setter
    def para_cruzamiento(self, para_cruzamiento):
        self._para_cruzamiento = para_cruzamiento

    @property
    def para_mutacion(self):
        return self._para_mutacion
    
    @para_mutacion.setter
    def para_mutacion(self, para_mutacion):
        self._para_mutacion = para_mutacion

    @property
    def cruzar_pares(self):
        return self._cruzar_pares
    
    @cruzar_pares.setter
    def cruzar_pares(self, cruzar_pares):
        self._cruzar_pares = cruzar_pares

    @property
    def criterio_reinicio(self):
        return self._criterio_reinicio
    
    @criterio_reinicio.setter
    def criterio_reinicio(self, criterio_reinicio):
        self._criterio_reinicio = criterio_reinicio
    
    @property
    def reinicio_poblacion(self):
        return self._reinicio_poblacion
    
    @reinicio_poblacion.setter
    def reinicio_poblacion(self, reinicio_poblacion):
        self._reinicio_poblacion = reinicio_poblacion

    @property
    def ratio_reinicio(self):
        return self._ratio_reinicio
    
    @ratio_reinicio.setter
    def ratio_reinicio(self, ratio_reinicio):
        self._ratio_reinicio = ratio_reinicio

    @property
    def paso_reinicio(self):
        return self._paso_reinicio
    
    @paso_reinicio.setter
    def paso_reinicio(self, paso_reinicio):
        self._paso_reinicio = paso_reinicio

    @property
    def mejor(self):
        return self._mejor
    
    @mejor.setter
    def mejor(self, mejor):
        self._mejor = mejor

    @property
    def maximizar(self):
        return self._maximizar
    
    @maximizar.setter
    def maximizar(self, maximizar):
        self._maximizar = maximizar
    
    @property
    def metodo_mutacion_elite(self):
        return self._metodo_mutacion_elite
    
    @metodo_mutacion_elite.setter
    def metodo_mutacion_elite(self, metodo_mutacion_elite):
        self._metodo_mutacion_elite = metodo_mutacion_elite

    @property
    def para_mutacion_elite(self):
        return self._para_mutacion_elite
    
    @para_mutacion_elite.setter
    def para_mutacion_elite(self, para_mutacion_elite):
        self._para_mutacion_elite = para_mutacion_elite
