
from pathlib import Path

import pandas as pd

import argparse

import mlflow

import mlflow.sklearn

import matplotlib.pyplot as plt

from mlflow.models import infer_signature

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import GroupShuffleSplit

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# Ruta predeterminada del conjunto de datos
DATA_PATH = Path(__file__).resolve().parent / "data" / "TUANDROMD.csv"


def load_data(file_path: Path) -> pd.DataFrame:
    """Carga el conjunto de datos original de malware Android."""

    # Verificar que el archivo exista antes de cargarlo
    if not file_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el conjunto de datos: {file_path}"
        )

    # Cargar el conjunto de datos desde el archivo CSV
    df = pd.read_csv(file_path)

    print(f"Dimensiones originales: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y prepara los datos para el entrenamiento del modelo."""

    # Crear una copia para conservar los datos originales
    df_clean = df.copy()

    # Eliminar únicamente las observaciones completamente vacías
    df_clean = df_clean.dropna(how="all").reset_index(drop=True)

    # Verificar que exista la columna objetivo
    if "Label" not in df_clean.columns:
        raise ValueError(
            "El conjunto de datos debe contener la columna 'Label'."
        )

    # Comprobar que no existan valores faltantes después de la limpieza
    if df_clean.isna().any().any():
        raise ValueError(
            "Se encontraron valores faltantes después de la limpieza."
        )

    # Validar que las características y etiquetas sean binarias
    if not df_clean.isin([0, 1]).all().all():
        raise ValueError(
            "Se esperaban únicamente valores binarios (0 o 1)."
        )

    # Convertir los datos a enteros de 8 bits para optimizar memoria
    df_clean = df_clean.astype("int8")

    # Mostrar información del conjunto de datos procesado
    print(f"Dimensiones después de la limpieza: {df_clean.shape}")
    print(f"Valores faltantes restantes: {df_clean.isna().sum().sum()}")
    print(f"Número de características predictoras: {df_clean.shape[1] - 1}")

    return df_clean

def split_data(df: pd.DataFrame, random_state: int = 42):
    """Divide los datos en entrenamiento y prueba utilizando grupos."""

    # Separar las características predictoras y las etiquetas
    # Conservar los tipos originales para generar grupos reproducibles
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # Identificar patrones únicos antes de convertir los tipos de datos
    # Esto conserva los mismos identificadores del experimento original
    groups = pd.util.hash_pandas_object(X, index=False)

    # Convertir las características a valores flotantes
    # para generar un esquema compatible con MLflow
    X = X.astype("float64")

    print(f"\nObservaciones totales: {len(X)}")
    print(f"Patrones únicos de características: {groups.nunique()}")

    # Configurar la división por grupos
    # El 20 % corresponde a grupos de prueba, no necesariamente
    # al 20 % exacto de las observaciones
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state
    )

    # Obtener los índices de entrenamiento y prueba
    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    # Separar los conjuntos mediante sus índices
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Verificar que no existan patrones compartidos
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    shared_groups = train_groups.intersection(test_groups)

    if shared_groups:
        raise ValueError(
            "Se encontraron patrones compartidos entre los conjuntos."
        )

    # Mostrar las dimensiones de los conjuntos resultantes
    print(f"\nObservaciones de entrenamiento: {len(X_train)}")
    print(f"Observaciones de prueba: {len(X_test)}")
    print(f"Patrones compartidos: {len(shared_groups)}")

    # Mostrar la distribución porcentual de las clases
    print("\nDistribución de clases - entrenamiento (%):")
    print(
        (y_train.value_counts(normalize=True).sort_index() * 100)
        .round(2)
    )

    print("\nDistribución de clases - prueba (%):")
    print(
        (y_test.value_counts(normalize=True).sort_index() * 100)
        .round(2)
    )

    return X_train, X_test, y_train, y_test

def train_model(
    X_train,
    y_train,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight=None,
    random_state: int = 42
):
    """
    Entrena un modelo de árbol de decisión para clasificar malware Android.

    Permite modificar los hiperparámetros del modelo y aplicar
    ponderación de clases para considerar el desbalance de datos.
    """

    # Inicializar el clasificador con los hiperparámetros definidos
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state
    )

    # Entrenar el modelo utilizando las observaciones de entrenamiento
    model.fit(X_train, y_train)

    # Mostrar información de la configuración utilizada
    print("\n--- Entrenamiento del modelo ---")
    print(f"Algoritmo: {type(model).__name__}")
    print(f"Profundidad máxima configurada: {max_depth}")
    print(f"Profundidad alcanzada: {model.get_depth()}")
    print(f"Número de hojas: {model.get_n_leaves()}")
    print(f"Muestras mínimas para división: {min_samples_split}")
    print(f"Muestras mínimas por hoja: {min_samples_leaf}")
    print(f"Ponderación de clases: {class_weight}")
    print(f"Semilla aleatoria: {random_state}")

    print("\nModelo entrenado correctamente.")

    return model


def evaluate_model(model, X_test, y_test):
    """Evalúa el desempeño del modelo utilizando el conjunto de prueba."""

    # Generar predicciones sobre las observaciones de prueba
    y_pred = model.predict(X_test)

    # Calcular las métricas principales de clasificación
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, y_pred
        )
    }

    # Mostrar los resultados de las métricas principales
    print("\n--- Evaluación del modelo ---")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Mostrar el desempeño individual de cada clase
    print("\n--- Reporte de clasificación ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Clase 0", "Clase 1"],
            zero_division=0
        )
    )

    # Calcular y mostrar la matriz de confusión
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("\n--- Matriz de confusión ---")
    print("Filas: clases reales | Columnas: clases predichas")
    print(matrix)

    return metrics, matrix


def save_confusion_matrix(matrix, output_dir: Path) -> Path:
    """Genera y guarda una representación visual de la matriz de confusión."""

    # Crear el directorio de artefactos si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Definir la ruta del archivo de imagen
    output_path = output_dir / "confusion_matrix.png"

    # Crear la visualización de la matriz de confusión
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Clase 0", "Clase 1"]
    )

    # Generar la figura
    fig, ax = plt.subplots(figsize=(6, 5))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    # Configurar los títulos y etiquetas de la figura
    ax.set_title("Matriz de confusión del modelo")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")

    # Guardar la imagen en el directorio de artefactos
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    # Liberar los recursos de la figura
    plt.close(fig)

    print(f"\nMatriz de confusión guardada en: {output_path}")

    return output_path

def parse_args():
    """Define y procesa los argumentos de la línea de comandos."""

    # Crear el analizador de argumentos
    parser = argparse.ArgumentParser(
        description="Entrenamiento y evaluación de malware Android."
    )

    # Configurar la profundidad máxima del árbol
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Profundidad máxima del árbol de decisión."
    )

    # Configurar el mínimo de muestras necesarias para dividir un nodo
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Número mínimo de muestras para dividir un nodo."
    )

    # Configurar el mínimo de muestras por hoja
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Número mínimo de muestras por hoja."
    )

    # Configurar la ponderación de clases
    parser.add_argument(
        "--class-weight",
        type=str,
        choices=["none", "balanced"],
        default="none",
        help="Ponderación de clases: none o balanced."
    )

    # Configurar la semilla para reproducibilidad
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para la división y el entrenamiento."
    )

    # Procesar y devolver los argumentos recibidos
    return parser.parse_args()


def main():
    """Ejecuta el entrenamiento con parámetros configurables."""

    # Obtener los argumentos de la línea de comandos
    args = parse_args()

    # Configurar el almacenamiento local de experimentos
    project_dir = Path(__file__).resolve().parent
    mlflow.set_tracking_uri(
        f"sqlite:///{project_dir / 'mlflow.db'}"
    )

    # Seleccionar el experimento donde se almacenarán las ejecuciones
    mlflow.set_experiment("TUANDROMD_DecisionTree")
    
    # Convertir la opción textual a un valor compatible con sklearn
    class_weight = (
        None if args.class_weight == "none" else "balanced"
    )

    # Cargar y preparar el conjunto de datos
    df = load_data(DATA_PATH)
    df_clean = preprocess_data(df)

    # Mostrar la distribución general de las clases
    print("\nDistribución general de las clases:")
    print(df_clean["Label"].value_counts().sort_index())

    # Separar los datos utilizando la semilla configurada
    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        random_state=args.random_state
    )

    # Entrenar el modelo con los hiperparámetros recibidos
    model = train_model(
        X_train,
        y_train,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=class_weight,
        random_state=args.random_state
    )

   
    # Iniciar una ejecución para registrar los resultados del modelo
    with mlflow.start_run():

        # Registrar los hiperparámetros utilizados
        mlflow.log_params({
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": args.class_weight,
            "random_state": args.random_state
        })

        # Evaluar el modelo con el conjunto de prueba
        metrics, matrix = evaluate_model(
            model,
            X_test,
            y_test
        )

        # Registrar las métricas de evaluación
        mlflow.log_metrics(metrics)

        # Generar y registrar la matriz de confusión como artefacto
        artifact_dir = project_dir / "artifacts"
        matrix_path = save_confusion_matrix(
            matrix,
            artifact_dir
        )

        mlflow.log_artifact(
            str(matrix_path),
            artifact_path="evaluacion"
        )

        # Inferir la firma del modelo a partir de los datos y predicciones
        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        # Registrar el modelo entrenado utilizando serialización segura
        # Autorizar únicamente el tipo interno del árbol de decisión
        # generado y controlado dentro de este experimento
        mlflow.sklearn.log_model(
            sk_model=model,
            name="decision_tree_model",
            signature=signature,
            input_example=X_train.head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # Mostrar el identificador de la ejecución
        print("\n--- Registro de MLflow ---")
        print(f"Identificador de ejecución: {mlflow.active_run().info.run_id}")
        print("Parámetros, métricas, artefactos y modelo registrados correctamente.")

    print("\nProceso de entrenamiento y evaluación completado.")

if __name__ == "__main__":
    main()

