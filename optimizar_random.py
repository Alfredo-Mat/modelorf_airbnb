import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CARGAR DATASET
# ============================================================

archivo = "airbnb_barcelona_modelo.csv"

df = pd.read_csv(
    archivo,
    sep=";",
    decimal=","
)

print("=" * 70)
print("CARGA DEL DATASET")
print("=" * 70)

print(f"Filas iniciales: {len(df):,}")


# ============================================================
# 2. ELIMINAR REGISTROS SIN PRECIO
# ============================================================

df = df.dropna(
    subset=["price"]
).copy()

print(
    f"Filas después de eliminar precios nulos: "
    f"{len(df):,}"
)


# ============================================================
# 3. IMPUTACIÓN
# ============================================================

variables_imputar = [
    "bedrooms",
    "beds",
    "bathrooms"
]

print("\n" + "=" * 70)
print("IMPUTACIÓN")
print("=" * 70)


for variable in variables_imputar:

    faltantes_antes = df[variable].isna().sum()

    # --------------------------------------------------------
    # 1. room_type + accommodates
    # --------------------------------------------------------

    mediana_grupo = (
        df.groupby(
            ["room_type", "accommodates"]
        )[variable]
        .transform("median")
    )

    df[variable] = df[variable].fillna(
        mediana_grupo
    )

    # --------------------------------------------------------
    # 2. accommodates
    # --------------------------------------------------------

    mediana_accommodates = (
        df.groupby("accommodates")[variable]
        .transform("median")
    )

    df[variable] = df[variable].fillna(
        mediana_accommodates
    )

    # --------------------------------------------------------
    # 3. Mediana global
    # --------------------------------------------------------

    df[variable] = df[variable].fillna(
        df[variable].median()
    )

    # --------------------------------------------------------
    # Redondeo
    # --------------------------------------------------------

    if variable in ["bedrooms", "beds"]:

        df[variable] = (
            df[variable]
            .round()
            .astype(int)
        )

    elif variable == "bathrooms":

        df[variable] = (
            df[variable]
            .round(1)
        )

    faltantes_despues = df[variable].isna().sum()

    print(
        f"{variable}: "
        f"{faltantes_antes:,} -> "
        f"{faltantes_despues:,} nulos"
    )


# ============================================================
# 4. VARIABLE OBJETIVO
# ============================================================

df["log_price"] = np.log1p(
    df["price"]
)


# ============================================================
# 5. VARIABLES
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
    variables_numericas +
    variables_categoricas
)

X = df[variables_modelo].copy()

y = df["log_price"].copy()


# ============================================================
# 6. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("DIVISIÓN DE DATOS")
print("=" * 70)

print(
    f"Entrenamiento: {len(X_train):,}"
)

print(
    f"Test:          {len(X_test):,}"
)


# ============================================================
# 7. PREPROCESAMIENTO
# ============================================================

preprocesamiento = ColumnTransformer(
    transformers=[

        (
            "numericas",
            "passthrough",
            variables_numericas
        ),

        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            variables_categoricas
        )
    ]
)


# ============================================================
# 8. RANDOM FOREST OPTIMIZADO
# ============================================================

modelo = Pipeline(
    steps=[

        (
            "preprocesamiento",
            preprocesamiento
        ),

        (
            "modelo",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 9. ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("ENTRENANDO RANDOM FOREST OPTIMIZADO")
print("=" * 70)

modelo.fit(
    X_train,
    y_train
)

print("Modelo entrenado.")


# ============================================================
# 10. OBTENER IMPORTANCIA DE LAS VARIABLES
# ============================================================

rf = modelo.named_steps["modelo"]

preprocesador = (
    modelo.named_steps["preprocesamiento"]
)

nombres_variables = (
    preprocesador
    .get_feature_names_out()
)

importancias = rf.feature_importances_


# ============================================================
# 11. IMPORTANCIA DE CADA VARIABLE TRANSFORMADA
# ============================================================

importancias_df = pd.DataFrame({

    "variable": nombres_variables,

    "importancia": importancias

})

importancias_df = (
    importancias_df
    .sort_values(
        "importancia",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. TOP 20 VARIABLES TRANSFORMADAS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 VARIABLES")
print("=" * 70)

print(
    importancias_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 13. AGRUPAR IMPORTANCIA POR VARIABLE ORIGINAL
# ============================================================

def variable_original(nombre):

    if nombre.startswith("numericas__"):
        return nombre.replace(
            "numericas__",
            ""
        )

    if nombre.startswith(
        "categoricas__"
    ):

        nombre = nombre.replace(
            "categoricas__",
            ""
        )

        # Nombre original antes del "__"
        return nombre.split("_")[0]

    return nombre


# ------------------------------------------------------------
# En lugar de intentar reconstruir nombres mediante split,
# utilizamos directamente los nombres originales comprobando
# qué variable contiene cada nombre transformado.
# ------------------------------------------------------------

def identificar_variable(nombre):

    nombre_limpio = nombre.replace(
        "numericas__",
        ""
    ).replace(
        "categoricas__",
        ""
    )

    for variable in variables_modelo:

        if nombre_limpio == variable:
            return variable

        if nombre_limpio.startswith(
            variable + "_"
        ):

            return variable

    return nombre_limpio


importancias_df[
    "variable_original"
] = importancias_df[
    "variable"
].apply(
    identificar_variable
)


# ============================================================
# 14. IMPORTANCIA AGRUPADA
# ============================================================

importancias_agrupadas = (
    importancias_df
    .groupby(
        "variable_original"
    )["importancia"]
    .sum()
    .sort_values(
        ascending=False
    )
    .reset_index()
)


# ============================================================
# 15. PORCENTAJE
# ============================================================

importancias_agrupadas[
    "importancia_pct"
] = (
    importancias_agrupadas[
        "importancia"
    ] * 100
)


# ============================================================
# 16. RESULTADO AGRUPADO
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANCIA POR VARIABLE ORIGINAL")
print("=" * 70)

print(
    importancias_agrupadas
    .to_string(index=False)
)


# ============================================================
# 17. COMPROBACIÓN
# ============================================================

print("\n" + "=" * 70)
print("COMPROBACIÓN")
print("=" * 70)

print(
    f"Suma importancias: "
    f"{importancias.sum():.6f}"
)


# ============================================================
# 18. TOP 10
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 VARIABLES DEL MODELO")
print("=" * 70)

for i, fila in enumerate(
    importancias_agrupadas.head(10).itertuples(),
    start=1
):

    print(
        f"{i:2d}. "
        f"{fila.variable_original:<35} "
        f"{fila.importancia_pct:.2f}%"
    )


print("\n" + "=" * 70)
print("PROCESO FINALIZADO")
print("=" * 70)