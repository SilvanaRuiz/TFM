import time
import pickle as pl
import pandas as pd
import numpy as np
import os
import googlemaps

# Constantes globales
BATCH_SIZE = 53
DEFAULT_TIME_MINUTES = 10000

def cargar_datos_pickle(filepath):
    """Carga datos desde un archivo pickle."""
    with open(filepath, 'rb') as f:
        return pl.load(f)

def asignar_coordenadas(dataframe, referencias, columna_busqueda, lat_col, lon_col):
    """Asigna coordenadas al dataframe basándose en referencias."""
    for i, fila in dataframe.iterrows():
        for referencia in referencias:
            if fila[columna_busqueda] == referencia[0]:
                dataframe.at[i, lat_col] = referencia[2]
                dataframe.at[i, lon_col] = referencia[3]
    return dataframe

def calcular_distancia_tiempo(api_key, lat1, lon1, lat2, lon2, max_retries=3):
    """Calcula la distancia y el tiempo entre dos coordenadas usando Google Maps."""
    gmaps = googlemaps.Client(key=api_key)
    origen = (lat1, lon1)
    destino = (lat2, lon2)
    
    for intento in range(max_retries):
        try:
            resultado = gmaps.distance_matrix(origen, destino, mode="driving")
            if resultado['rows'][0]['elements'][0]['status'] == 'OK':
                distancia = resultado['rows'][0]['elements'][0]['distance']['text']
                tiempo = resultado['rows'][0]['elements'][0]['duration']['text']
                return distancia, tiempo
        except Exception as e:
            print(f"Error en el intento {intento + 1}: {e}")
            if intento < max_retries - 1:
                time.sleep(5)
            else:
                return "Error", "Error"

def localizacion_centro_provincia(dataframe_principal):
    """Calcula la lejanía entre centros y municipios."""
    centros_provincias = dataframe_principal[['NOMBRE_CENTRO', 'PROVINCIA_RESIDENCIA']].copy()

    # Cargar datos desde archivos pickle
    prison_adress = cargar_datos_pickle('/Users/silvanaruizmedina/Desktop/TFM/Eventos/objectos/prison_adresses_nuevo.pickle')
    municipio_adress = cargar_datos_pickle('/Users/silvanaruizmedina/Desktop/TFM/Eventos/objectos/municipio_adresses_nuevo.pickle')
    distancia_tiempo_dict = cargar_datos_pickle('/Users/silvanaruizmedina/Desktop/TFM/Eventos/objectos/distancia_tiempo_dict_nuevo.pickle')
    
    # Inicializar coordenadas
    for col in ['NOMBRE_CENTRO_LAT', 'NOMBRE_CENTRO_LON', 'PROVINCIA_RESIDENCIA_LAT', 'PROVINCIA_RESIDENCIA_LON']:
        centros_provincias[col] = None

    # Asignar coordenadas
    centros_provincias = asignar_coordenadas(centros_provincias, prison_adress, 'NOMBRE_CENTRO', 'NOMBRE_CENTRO_LAT', 'NOMBRE_CENTRO_LON')
    centros_provincias = asignar_coordenadas(centros_provincias, municipio_adress, 'PROVINCIA_RESIDENCIA', 'PROVINCIA_RESIDENCIA_LAT', 'PROVINCIA_RESIDENCIA_LON')

    # Inicializar columnas de distancia y tiempo
    centros_provincias['DISTANCIA'] = None
    centros_provincias['TIEMPO'] = None
   

    # Leer la clave API desde las variables de entorno
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        raise ValueError("Clave API de Google no encontrada en el entorno virtual.")

    # Procesar en lotes
    total_batches = len(centros_provincias) // BATCH_SIZE + (1 if len(centros_provincias) % BATCH_SIZE != 0 else 0)
    for start in range(0, len(centros_provincias), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(centros_provincias))
        batch = centros_provincias.iloc[start:end]
        
        for i, row in batch.iterrows():
            clave = (row['NOMBRE_CENTRO'], row['PROVINCIA_RESIDENCIA'])
            
            if clave in distancia_tiempo_dict:
                centros_provincias.at[i, 'DISTANCIA'] = distancia_tiempo_dict[clave]['DISTANCIA']
                centros_provincias.at[i, 'TIEMPO'] = distancia_tiempo_dict[clave]['TIEMPO']
            else:
                #coords = [row['NOMBRE_CENTRO_LAT'], row['NOMBRE_CENTRO_LON'], row['PROVINCIA_RESIDENCIA_LAT'], row['PROVINCIA_RESIDENCIA_LON']]
                #if pd.notna(coords).all():
                    #distancia, tiempo = calcular_distancia_tiempo(api_key, *coords)
                    #distancia_tiempo_dict[clave] = {'DISTANCIA': distancia, 'TIEMPO': tiempo}
                    #centros_provincias.at[i, 'DISTANCIA'] = distancia
                    #centros_provincias.at[i, 'TIEMPO'] = tiempo
                coords = [
        row['NOMBRE_CENTRO_LAT'],
        row['NOMBRE_CENTRO_LON'],
        row['PROVINCIA_RESIDENCIA_LAT'],
        row['PROVINCIA_RESIDENCIA_LON']]

                if all(c is not None and pd.notna(c) for c in coords):
                    distancia, tiempo = calcular_distancia_tiempo(api_key, *coords)
                else:
                    print(f"Coordenadas inválidas para centro '{row['NOMBRE_CENTRO']}' y provincia '{row['PROVINCIA_RESIDENCIA']}': {coords}")
                    distancia, tiempo = "Error", "Error"
                distancia, tiempo = calcular_distancia_tiempo(api_key, *coords)
                distancia_tiempo_dict[clave] = {'DISTANCIA': distancia, 'TIEMPO': tiempo}
                centros_provincias.at[i, 'DISTANCIA'] = distancia
                centros_provincias.at[i, 'TIEMPO'] = tiempo

        # Convertir tiempo a minutos
        centros_provincias['TIEMPO'] = pd.to_timedelta(centros_provincias['TIEMPO'].str.replace('mins', 'min'), errors='coerce').dt.total_seconds() / 60
        centros_provincias['TIEMPO'] = centros_provincias['TIEMPO'].fillna(DEFAULT_TIME_MINUTES)

        # Asignar al DataFrame original
        dataframe_principal['LEJANIA_CENTRO_MUNICIPIO'] = centros_provincias['TIEMPO']

        # Guardar distancias actualizadas
        with open('/Users/silvanaruizmedina/Desktop/TFM/Eventos/objectos/distancia_tiempo_dict.pickle', 'wb') as f:
            pl.dump(distancia_tiempo_dict, f)
        print("Diccionario de distancias y tiempos actualizado y guardado.")

        return dataframe_principal

# Función para convertir Años-Meses-Días a días
def convertir_a_dias(valor):
    """
    Convierte una cadena en formato 'Años-Meses-Días' a días totales.

    Args:
        valor (str): Cadena en formato 'Años-Meses-Días'.

    Returns:
        int: Total de días calculados.
        0: Si el valor es NaN o no válido.
    """
    try:
        # Devolver valores nulos con 0 (desconocido¬todavia no condenado)
        if pd.isna(valor):
            return 0
        
        # Dividir el valor en años, meses y días
        años, meses, dias = map(int, valor.split('-'))
        
        # Convertir todo a días
        total_dias = (años * 365) + (meses * 30) + dias
        return total_dias
    
    except (ValueError, AttributeError):
        # Si hay un error de formato o tipo, devolver 0 en lugar de None
        return 0
