# Clasificación de malware Android con MLflow

## Descripción del proyecto

Este proyecto desarrolla un flujo reproducible de experimentación de aprendizaje automático para la clasificación de malware Android utilizando el conjunto de datos **TUNADROMD**, disponible en el UCI Machine Learning Repository.

Se utiliza el algoritmo `DecisionTreeClassifier` de scikit-learn y **MLflow** para registrar hiperparámetros, métricas de evaluación, modelos entrenados y artefactos.

El proyecto forma parte de la actividad de experimentación de la materia de MLOps de la Maestría en Inteligencia Artificial Aplicada del Tecnológico de Monterrey.

## Tecnologías utilizadas

- Python 3.14
- scikit-learn
- pandas y NumPy
- Matplotlib
- MLflow 3
- Git y GitHub
- Ubuntu mediante Windows Subsystem for Linux (WSL)
- Visual Studio Code

Las versiones exactas de las dependencias se encuentran en `requirements.txt`.

## Repositorio

**GitHub:** https://github.com/josafaatr1/MLOps

## Conjunto de datos

**Nombre:** TUNADROMD

**Fuente:** UCI Machine Learning Repository

**Enlace:** https://archive.ics.uci.edu/dataset/813/tunadromd

El conjunto de datos contiene originalmente 4,465 observaciones y 241 características predictoras binarias.

Durante el preprocesamiento:

1. Se elimina una observación completamente vacía.
2. Se verifica la ausencia de valores faltantes.
3. Se validan las características y etiquetas binarias.
4. Se identifican patrones de características repetidos.
5. Se separan los datos mediante `GroupShuffleSplit` para evitar que patrones idénticos aparezcan simultáneamente en entrenamiento y prueba.

Después de la limpieza se obtienen **4,464 observaciones y 660 patrones únicos**.

### Distribución de los datos

| Conjunto | Observaciones |
|---|---:|
| Entrenamiento | 2,903 |
| Prueba | 1,561 |
| Total | 4,464 |

No existen patrones de características compartidos entre ambos conjuntos.

## Estructura del proyecto

```text
mlops-actividad3/
├── data/
│   └── TUANDROMD.csv
├── notebooks/
├── artifacts/
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

Los archivos de datos, los modelos y los artefactos generados localmente pueden excluirse del control de versiones mediante `.gitignore`.

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/josafaatr1/MLOps.git
cd MLOps
```

Si el proyecto se encuentra en un subdirectorio del repositorio, acceder a dicho directorio antes de continuar.

### 2. Crear el entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

### 4. Preparar el conjunto de datos

Descargar TUNADROMD desde UCI y colocar el archivo `TUANDROMD.csv` en:

```text
data/TUANDROMD.csv
```

## Ejecución del modelo

### Experimento base

```bash
python train.py --max-depth 5 --min-samples-split 2 --min-samples-leaf 1 --class-weight none --random-state 42
```

### Configuración personalizada

```bash
python train.py --max-depth 3 --min-samples-split 2 --min-samples-leaf 1 --class-weight none --random-state 42
```

### Consultar argumentos disponibles

```bash
python train.py --help
```

Los argumentos permiten modificar la profundidad máxima, el número mínimo de muestras para dividir nodos, el número mínimo de muestras por hoja, la ponderación de clases y la semilla aleatoria.

## Seguimiento de experimentos con MLflow

El script registra automáticamente:

- Cinco parámetros de configuración.
- Seis métricas: `accuracy`, `precision`, `recall`, `f1_score`, `f1_macro` y `balanced_accuracy`.
- El modelo entrenado de scikit-learn.
- La matriz de confusión como artefacto PNG.

El nombre del experimento es:

`TUANDROMD_DecisionTree`

### Iniciar la interfaz de MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir en el navegador:

http://localhost:5000

## Configuraciones experimentales

Se ejecutaron seis configuraciones del árbol de decisión, manteniendo la misma partición de datos y la semilla aleatoria 42.

| Experimento | max_depth | min_samples_split | min_samples_leaf | class_weight |
|---|---:|---:|---:|---|
| E1 — Base | 5 | 2 | 1 | none |
| E2 — Mayor profundidad | 10 | 2 | 1 | none |
| E3 — Menor profundidad | 3 | 2 | 1 | none |
| E4 — División restringida | 5 | 10 | 1 | none |
| E5 — Hojas más grandes | 5 | 2 | 5 | none |
| E6 — Clases ponderadas | 5 | 2 | 1 | balanced |

## Resultados experimentales

| Experimento | Accuracy | F1-macro | Balanced accuracy |
|---|---:|---:|---:|
| E1 | 95.90% | 90.00% | 94.67% |
| E2 | 96.99% | 92.49% | **96.38%** |
| E3 | **98.14%** | **94.82%** | 93.71% |
| E4 | 95.90% | 90.00% | 94.67% |
| E5 | 95.32% | 88.68% | 93.52% |
| E6 | 95.58% | 89.50% | 95.32% |

### Análisis de resultados

El experimento **E3**, con profundidad máxima de 3, obtuvo el mejor rendimiento en exactitud y F1-macro. Esto indica que un árbol de menor complejidad puede generalizar mejor que configuraciones más profundas en la partición evaluada.

Por otro lado, **E2**, con profundidad máxima de 10, obtuvo la mayor exactitud balanceada, mostrando un mejor equilibrio entre la sensibilidad de ambas clases.

La ponderación de clases mejoró la exactitud balanceada respecto al experimento base, pero no mejoró el F1-macro.

Estos resultados evidencian que la selección del mejor modelo depende de las métricas y del costo asociado a los distintos tipos de error de clasificación.

## Reproducibilidad

Para reproducir los experimentos:

1. Utilizar el mismo archivo del conjunto de datos.
2. Instalar las dependencias de `requirements.txt`.
3. Mantener `random_state=42`.
4. Ejecutar las seis configuraciones documentadas.
5. Comparar los resultados en MLflow.

La separación por grupos evita la presencia de patrones idénticos en los conjuntos de entrenamiento y prueba.

**Limitación metodológica:** los resultados corresponden a una única partición de entrenamiento y prueba. Para evaluar con mayor rigor la capacidad de generalización y seleccionar hiperparámetros sin utilizar el conjunto de prueba, se recomienda incorporar validación cruzada por grupos y una evaluación final independiente.

## Referencias

UCI Machine Learning Repository. (s. f.). *TUNADROMD*. https://archive.ics.uci.edu/dataset/813/tunadromd

MLflow. (s. f.). *MLflow documentation*. https://mlflow.org/docs/latest/

Scikit-learn developers. (s. f.). *DecisionTreeClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
