import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ARCHIVO = "airbnb_barcelona_modelo.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

MIN_OBSERVACIONES = 10


# ============================================================
# 2. CARGAR DATASET
# ============================================================

df = pd.read_csv(
    ARCHIVO,
    sep=";",
    decimal=","
)

print("==========================================")
print("DATASET")
print("==========================================")

print(f"Registros iniciales: {len(df):,}")


# ============================================================
# 3. ELIMINAR REGISTROS SIN PRECIO
# ============================================================

df = df[df["price"].notna()].copy()

print(f"Registros con precio: {len(df):,}")


# ============================================================
# 4. ASEGURAR TIPOS NUMÉRICOS
# ============================================================

numericas = [
    "latitude",
    "longitude",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "price"
]

for columna in numericas:

    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )


# ============================================================
# 5. IMPUTACIÓN DE BEDROOMS, BEDS Y BATHROOMS
# ============================================================

def imputar_variable(df, variable):

    resultado = df[variable].copy()

    # --------------------------------------------------------
    # NIVEL 1:
    # room_type + accommodates
    # --------------------------------------------------------

    grupos = (
        df.groupby(
            ["room_type", "accommodates"],
            dropna=False
        )[variable]
        .agg(
            count="count",
            median="median"
        )
    )

    grupos_validos = grupos[
        grupos["count"] >= MIN_OBSERVACIONES
    ]

    for indice, fila in grupos_validos.iterrows():

        room_type = indice[0]
        accommodates = indice[1]

        mascara = (
            resultado.isna()
            & (df["room_type"] == room_type)
            & (df["accommodates"] == accommodates)
        )

        resultado.loc[mascara] = fila["median"]

    # --------------------------------------------------------
    # NIVEL 2:
    # accommodates
    # --------------------------------------------------------

    grupos_acc = (
        df.groupby("accommodates")[variable]
        .agg(
            count="count",
            median="median"
        )
    )

    grupos_acc_validos = grupos_acc[
        grupos_acc["count"] >= MIN_OBSERVACIONES
    ]

    for accommodates, fila in grupos_acc_validos.iterrows():

        mascara = (
            resultado.isna()
            & (df["accommodates"] == accommodates)
        )

        resultado.loc[mascara] = fila["median"]

    # --------------------------------------------------------
    # NIVEL 3:
    # MEDIANA GLOBAL
    # --------------------------------------------------------

    mediana_global = df[variable].median()

    resultado = resultado.fillna(mediana_global)

    # --------------------------------------------------------
    # REDONDEO
    # --------------------------------------------------------

    if variable in ["bedrooms", "beds"]:

        resultado = resultado.round()

    elif variable == "bathrooms":

        resultado = resultado.round(1)

    return resultado


for variable in [
    "bedrooms",
    "beds",
    "bathrooms"
]:

    df[variable] = imputar_variable(
        df,
        variable
    )


# ============================================================
# 6. COMPROBAR NULOS
# ============================================================

print("\n==========================================")
print("NULOS DESPUÉS DE IMPUTACIÓN")
print("==========================================")

print(
    df[
        [
            "latitude",
            "longitude",
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "price"
        ]
    ]
    .isna()
    .sum()
)


# ============================================================
# 7. CREAR VARIABLE OBJETIVO
# ============================================================

df["log_price"] = np.log1p(
    df["price"]
)


# ============================================================
# 8. VARIABLES DEL MODELO
# ============================================================

variables_numericas = [
    "latitude",
    "longitude",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms"
]

variables_categoricas = [
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type"
]

variables_modelo = (
    variables_numericas
    + variables_categoricas
)

X = df[variables_modelo]

y = df["log_price"]


# ============================================================
# 9. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\n==========================================")
print("TRAIN / TEST")
print("==========================================")

print(f"Train: {len(X_train):,}")
print(f"Test:  {len(X_test):,}")

print(
    f"Porcentaje Train: "
    f"{len(X_train) / len(X) * 100:.1f}%"
)

print(
    f"Porcentaje Test:  "
    f"{len(X_test) / len(X) * 100:.1f}%"
)


# ============================================================
# 10. ONE-HOT ENCODING
# ============================================================

preprocesamiento_ridge = ColumnTransformer(

    transformers=[

        (
            "numericas",
            StandardScaler(),
            variables_numericas
        ),

        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            variables_categoricas
        )
    ]
)


# Para Random Forest y Gradient Boosting
# no necesitamos escalar las variables numéricas.

preprocesamiento_arboles = ColumnTransformer(

    transformers=[

        (
            "numericas",
            "passthrough",
            variables_numericas
        ),

        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            variables_categoricas
        )
    ]
)


# ============================================================
# 11. DEFINIR MODELOS
# ============================================================

modelo_ridge = Pipeline(

    steps=[

        (
            "preprocesamiento",
            preprocesamiento_ridge
        ),

        (
            "modelo",
            Ridge(alpha=10.0)
        )
    ]
)


modelo_random_forest = Pipeline(

    steps=[

        (
            "preprocesamiento",
            preprocesamiento_arboles
        ),

        (
            "modelo",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


modelo_gradient_boosting = Pipeline(

    steps=[

        (
            "preprocesamiento",
            preprocesamiento_arboles
        ),

        (
            "modelo",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ============================================================
# 12. FUNCIÓN DE EVALUACIÓN
# ============================================================

def evaluar_modelo(nombre, modelo):

    print("\n------------------------------------------")
    print(nombre)
    print("------------------------------------------")

    # Entrenamiento
    modelo.fit(
        X_train,
        y_train
    )

    # Predicción
    predicciones_log = modelo.predict(
        X_test
    )

    # Métricas en escala log
    mae_log = mean_absolute_error(
        y_test,
        predicciones_log
    )

    rmse_log = np.sqrt(
        mean_squared_error(
            y_test,
            predicciones_log
        )
    )

    r2 = r2_score(
        y_test,
        predicciones_log
    )

    # Volver a euros
    y_test_euros = np.expm1(
        y_test
    )

    predicciones_euros = np.expm1(
        predicciones_log
    )

    # Evitar precios negativos
    predicciones_euros = np.maximum(
        predicciones_euros,
        0
    )

    mae_euros = mean_absolute_error(
        y_test_euros,
        predicciones_euros
    )

    rmse_euros = np.sqrt(
        mean_squared_error(
            y_test_euros,
            predicciones_euros
        )
    )

    print(f"MAE log:   {mae_log:.4f}")
    print(f"RMSE log:  {rmse_log:.4f}")
    print(f"R²:        {r2:.4f}")

    print("\nEscala euros:")
    print(f"MAE:       {mae_euros:.2f} €")
    print(f"RMSE:      {rmse_euros:.2f} €")

    return {
        "modelo": nombre,
        "MAE_log": mae_log,
        "RMSE_log": rmse_log,
        "R2": r2,
        "MAE_euros": mae_euros,
        "RMSE_euros": rmse_euros
    }


# ============================================================
# 13. ENTRENAR MODELOS
# ============================================================

resultados = []

resultados.append(
    evaluar_modelo(
        "Ridge Regression",
        modelo_ridge
    )
)

resultados.append(
    evaluar_modelo(
        "Random Forest",
        modelo_random_forest
    )
)

resultados.append(
    evaluar_modelo(
        "Gradient Boosting",
        modelo_gradient_boosting
    )
)


# ============================================================
# 14. COMPARACIÓN FINAL
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)

resultados_df = resultados_df.sort_values(
    by="MAE_euros"
)

print("\n==========================================")
print("COMPARACIÓN FINAL")
print("==========================================")

print(
    resultados_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. MEJOR MODELO
# ============================================================

mejor = resultados_df.iloc[0]

print("\n==========================================")
print("MEJOR MODELO SEGÚN MAE")
print("==========================================")

print(
    f"Modelo: {mejor['modelo']}"
)

print(
    f"MAE: {mejor['MAE_euros']:.2f} €"
)

print(
    f"RMSE: {mejor['RMSE_euros']:.2f} €"
)

print(
    f"R²: {mejor['R2']:.4f}"
)

print("\nNo se ha guardado ningún archivo.")