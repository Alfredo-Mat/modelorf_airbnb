import pandas as pd

# ==========================================
# 1. CARGAR ARCHIVOS
# ==========================================

calendar_weather = pd.read_csv(
    "calendar_barcelona_weather_real.csv"
)

listings = pd.read_csv(
    "listings.csv"
)

print("==========================================")
print("ARCHIVOS CARGADOS")
print("==========================================")

print("Calendar + weather:", len(calendar_weather), "filas")
print("Listings:", len(listings), "filas")


# ==========================================
# 2. COMPROBAR TIPOS DE ID
# ==========================================

print("\nTipos de ID:")
print("calendar listing_id:", calendar_weather["listing_id"].dtype)
print("listings id:", listings["id"].dtype)


# ==========================================
# 3. ASEGURAR MISMO TIPO
# ==========================================

calendar_weather["listing_id"] = pd.to_numeric(
    calendar_weather["listing_id"],
    errors="coerce"
)

listings["id"] = pd.to_numeric(
    listings["id"],
    errors="coerce"
)


# ==========================================
# 4. COMPROBAR COINCIDENCIAS
# ==========================================

ids_calendar = set(
    calendar_weather["listing_id"].dropna().unique()
)

ids_listings = set(
    listings["id"].dropna().unique()
)

ids_comunes = ids_calendar.intersection(ids_listings)

print("\n==========================================")
print("COINCIDENCIA DE IDs")
print("==========================================")

print("Listings únicos en calendar:", len(ids_calendar))
print("Listings únicos en listings:", len(ids_listings))
print("IDs coincidentes:", len(ids_comunes))

print(
    "Porcentaje de listings del calendar con correspondencia:",
    round(len(ids_comunes) / len(ids_calendar) * 100, 2),
    "%"
)


# ==========================================
# 5. HACER MERGE
# ==========================================

dataset_final = calendar_weather.merge(
    listings,
    left_on="listing_id",
    right_on="id",
    how="left",
    suffixes=("", "_listing")
)


# ==========================================
# 6. COMPROBAR RESULTADO
# ==========================================

print("\n==========================================")
print("RESULTADO DEL MERGE")
print("==========================================")

print("Filas:", len(dataset_final))
print("Columnas:", len(dataset_final.columns))

print("\nFilas originales calendar + weather:",
      len(calendar_weather))

print("Filas después del merge:",
      len(dataset_final))


# ==========================================
# 7. COMPROBAR LISTINGS SIN MATCH
# ==========================================

sin_match = dataset_final["id"].isna().sum()

print("\nListings sin correspondencia:")
print(sin_match)

print(
    "Porcentaje de filas sin correspondencia:",
    round(sin_match / len(dataset_final) * 100, 2),
    "%"
)


# ==========================================
# 8. MOSTRAR ALGUNAS COLUMNAS
# ==========================================

print("\n==========================================")
print("EJEMPLO")
print("==========================================")

columnas_mostrar = [
    "listing_id",
    "date",
    "available",
    "temp",
    "prcp",
    "rain"
]

# Añadir algunas columnas de listings si existen
for columna in ["name", "neighbourhood", "room_type", "price"]:
    if columna in dataset_final.columns:
        columnas_mostrar.append(columna)

print(
    dataset_final[columnas_mostrar].head(10)
)


# ==========================================
# 9. GUARDAR
# ==========================================

salida = "airbnb_barcelona_calendar_weather_listings.csv"

dataset_final.to_csv(
    salida,
    index=False
)

print("\n==========================================")
print("ARCHIVO GENERADO")
print("==========================================")

print(salida)
# Listings del calendar que NO aparecen en listings.csv

ids_sin_match = sorted(
    ids_calendar - ids_listings
)

print("Número de IDs sin match:", len(ids_sin_match))

print("\nPrimeros 30 IDs sin match:")
print(ids_sin_match[:30])

print("\nFilas del calendar correspondientes:")
print(
    calendar_weather[
        calendar_weather["listing_id"].isin(ids_sin_match)
    ]
    .groupby("listing_id")
    .size()
    .sort_values(ascending=False)
    .head(20)
)