#Biblioteca
import datetime as dt
from datetime import datetime
from dateutil import parser
from sklearn.preprocessing import OneHotEncoder
import pickle as pl 
import numpy as np
import re
import pandas as pd 
from category_encoders import BinaryEncoder
from calcular_distancia import localizacion_centro_provincia, cargar_datos_pickle, convertir_a_dias


def procesado_datos(df):
    # Resolver inconsistencias nombre col
    df.columns = df.columns.str.replace(' ', '_')

    # Eliminamos las filas que contengan vigente en PPS
    df = df.loc[~df['ULT_BAJA_PPS'].astype(str).fillna('').eq('Vigente')].copy()

    one_hot = OneHotEncoder()
 
    # Identificar las nacionalidades con frecuencia menor o igual a 20
    nacionalidades_poco_frecuentes = df['NACIONALIDAD'].value_counts()
    nacionalidades_poco_frecuentes = nacionalidades_poco_frecuentes[nacionalidades_poco_frecuentes <= 20].index
    
    # Tratamos posibles NaNs
    df['NACIONALIDAD'] = df['NACIONALIDAD'].fillna('Desconocido')

    # Reemplazar las nacionalidades poco frecuentes por 'Otros'
    df['NACIONALIDAD'] = df['NACIONALIDAD'].replace(nacionalidades_poco_frecuentes, 'Otros')
    #Codificamos 
    nacionalidad_encoded = one_hot.fit_transform(df[['NACIONALIDAD']]).toarray()
    nacionalidad_columns = one_hot.get_feature_names_out(['NACIONALIDAD'])
    encoded_nacionalidad = pd.DataFrame(nacionalidad_encoded, columns=nacionalidad_columns)

    # Llamar a la función para calcular la distancia
    df = localizacion_centro_provincia(dataframe_principal=df)

    # Encoding nombre_centro 
    nombre_centro_encoded = one_hot.fit_transform(df[['NOMBRE_CENTRO']]).toarray() 
    nombre_centro_columns = one_hot.get_feature_names_out(['NOMBRE_CENTRO'])
    encoded_nombre_centro = pd.DataFrame(nombre_centro_encoded, columns= nombre_centro_columns)

    # Encoding provincia_residencia
    provincia_residencia_encoded = one_hot.fit_transform(df[['PROVINCIA_RESIDENCIA']]).toarray()
    provincia_residencia_columns = one_hot.get_feature_names_out(['PROVINCIA_RESIDENCIA'])
    encoded_provincia_residencia = pd.DataFrame(provincia_residencia_encoded, columns=provincia_residencia_columns)

    # Convertir FECHA_NACIMIENTO a tipo datetime y extraer el año para usarla como variable del modelo
    df['AÑO_NACIMIENTO'] = pd.to_datetime(df['FECHA_NACIMIENTO'], errors='coerce').dt.year.astype('Int64')
   
    anio_actual = datetime.today().year

    # Calcular la edad restando el año de nacimiento al año actual
    df['EDAD'] = anio_actual - df['AÑO_NACIMIENTO']

    # Encoding sexo
    sex_encoder = one_hot.fit_transform(df[['SEXO']].fillna('Desconocido')).toarray()  # Manejo de nulos
    sex_columns = one_hot.get_feature_names_out(['SEXO'])  # Nombres de las columnas codificadas
    encoded_sex = pd.DataFrame(sex_encoder, columns=sex_columns)

    # Convertir las columnas de fecha a tipo datetime para cálculos de tiempo posteriores
    df['FECHA_ULTIMO_INGRESO'] = pd.to_datetime(df['FECHA_ULTIMO_INGRESO'], errors='coerce')
    df['FECHA_INGRESO_CENTRO_ACTUAL'] = pd.to_datetime(df['FECHA_INGRESO_CENTRO_ACTUAL'], errors='coerce')

    # Calcular diferencia en días
    df['PERIODO_ENTRE_INGRESOS'] = (df['FECHA_INGRESO_CENTRO_ACTUAL'] - df['FECHA_ULTIMO_INGRESO']).dt.days

    # Reemplazar nulos con 9900 dado que si hay NaN se entiende que no hay ingreso previo
    df['PERIODO_ENTRE_INGRESOS'] = df['PERIODO_ENTRE_INGRESOS'].fillna(9900).astype(int)

    # Encoding estado civil, suponemos NaN son no consta
    df['ESTADO_CIVIL']=df['ESTADO_CIVIL'].fillna('No consta')
    # Encoding estado civil
    estado_civil_encoding = one_hot.fit_transform(df[['ESTADO_CIVIL']]).toarray()
    estado_civil_columns = one_hot.get_feature_names_out(['ESTADO_CIVIL'])
    encoded_estado_civil = pd.DataFrame(estado_civil_encoding, columns=estado_civil_columns)

    #Cargamos el diccionario con el encdoing y lo ejecutamos 
    encoding_nivel_estudios = cargar_datos_pickle('/Users/silvanaruizmedina/Desktop/TFM/Eventos/objectos/encoding_nivel_estudios.pickle')
    df['NIVEL_ESTUDIOS'] = df['NIVEL_ESTUDIOS'].map(encoding_nivel_estudios)
    df['NIVEL_ESTUDIOS']= df['NIVEL_ESTUDIOS'].fillna(0)

    # Convertir los valores de la columna 'CONDENA_AAMMDD' a días totales en lugar de fechas
    df['CONDENA_DIAS'] = df['CONDENA_AAMMDD'].apply(convertir_a_dias)
    df['CONDENA_DIAS'] = df['CONDENA_DIAS'].astype(int)

    # Encoding de situación_penal
    situacion_penal = one_hot.fit_transform(df[['SITUACION_PENAL_INTERNO']]).toarray()
    situacion_penal_columns = one_hot.get_feature_names_out(['SITUACION_PENAL_INTERNO'])
    situacion_penal = pd.DataFrame(situacion_penal, columns= situacion_penal_columns)

    # Adaptarse al comportamiento futuro de la biblioteca pandas
    pd.set_option('future.no_silent_downcasting', True)
    # Encoding de CT_GRADO
    df['CT_GRADO'] = df['CT_GRADO'].replace({'SC': 40}).astype(int)
    
     # Diccionario de codificación
    encoding_ct_grado = {
        0: 0,
        10: 1,
        20: 2,
        30: 3,
        40: 4
    }
    df['CT_GRADO'] = df['CT_GRADO'].map(encoding_ct_grado)

    # Encoding TIPO_MODULO
    tipo_modulo_encoding = one_hot.fit_transform(df[['TIPO_MODULO_UBICACION_INTERNO']]).toarray()
    tipo_modulo_columns = one_hot.get_feature_names_out(['TIPO_MODULO_UBICACION_INTERNO'])
    tipo_modulo = pd.DataFrame(tipo_modulo_encoding, columns=tipo_modulo_columns)

    # Encoding de Acompañamiento celda
    df['ACOMPANADO_CELDA']= df['ACOMPANADO_CELDA'].fillna('No')
    acompañado_celda_encoding= one_hot.fit_transform(df[['ACOMPANADO_CELDA']]).toarray()
    acompañado_celda_columnas = one_hot.get_feature_names_out(['ACOMPANADO_CELDA'])
    acompañado_celda = pd.DataFrame(acompañado_celda_encoding, columns=acompañado_celda_columnas)


    # Encoding Delito_mayor
    binary_econding = BinaryEncoder()
    delito_encoding = pd.DataFrame(binary_econding.fit_transform(df['DELITO_MAYOR_CONDENA']))
        
    # Verificar si la columna es de tipo string y aplicar replace solo en ese caso
    if df['SALDO_PECULIO_ACTUAL'].dtype == object:
        # Reemplazar comas por puntos solo si la columna es de tipo string
        df['SALDO_PECULIO_ACTUAL'] = df['SALDO_PECULIO_ACTUAL'].str.replace(',', '.')

    # NaN=0, coomo si no recibiesen
    df['SALDO_PECULIO_ACTUAL']=df['SALDO_PECULIO_ACTUAL'].fillna(0).astype(float)

     # Faltas cumplir, contamos las totales y separamos por gravedad de la falta
    df['FALTAS_PENDIENTES_CUMPLIR']= df['FALTAS_PENDIENTES_CUMPLIR'].fillna(0)
    df['FALTAS_PENDIENTES_CUMPLIR'] = df['FALTAS_PENDIENTES_CUMPLIR'].apply(lambda x: re.sub(r'[A-Za-z]', '', str(x)))
    df['TOTAL_FALTAS_CUMPLIR'] = df['FALTAS_PENDIENTES_CUMPLIR'].apply(lambda x: len(x.split('-')) if x!= '0' else 0)
    df['NUMERO_FALTAS_LEVES'] = df['FALTAS_PENDIENTES_CUMPLIR'].apply(lambda x: x.count('110'))
    df['NUMERO_FALTAS_GRAVES'] = df['FALTAS_PENDIENTES_CUMPLIR'].apply(lambda x: x.count('109'))
    df['NUMERO_FALTAS_MUYGRAVES'] = df['FALTAS_PENDIENTES_CUMPLIR'].apply(lambda x: x.count('108'))

    # Seperar el tipo de autolesiones para cuantificar cuantas de cada tipo tiene
    df['TIPO_AUTOLESIONES_TOTAL'] = df['TIPO_AUTOLESIONES_ULT_12_MESES'].fillna('') + ' - ' + df['TIPO_AUTOLESIONES_ANT_12_MESES']
    df['TIPO_AUTOLESIONES_TOTAL'] = df['TIPO_AUTOLESIONES_TOTAL'].fillna('0')
    df['TIPO_AUTOLESIONES_TOTAL'] = df['TIPO_AUTOLESIONES_TOTAL'].str.replace(r'\s*-\s*', ' - ').str.strip()
    df['NUM_AUTOLESION_LEVE'] = df['TIPO_AUTOLESIONES_TOTAL'].apply(lambda x: x.count('Autolesión leve'))
    df['NUM_AUTOLESION_GRAVE'] = df['TIPO_AUTOLESIONES_TOTAL'].apply(lambda x: x.count('Autolesión grave'))
    df['NUM_AUTOLESION_MUY_GRAVE'] = df['TIPO_AUTOLESIONES_TOTAL'].apply(lambda x: x.count('Autolesión muy grave'))
    df['AUTOLESIONES_TOTAL'] = (df['AUTOLESIONES_ANTERIOR_12_MESES']) +  (df['AUTOLESIONES_ULT_12_MESES'])
    df['INTENTO_SUICIDIO'] = (df['INTENTO_SUICIDIO_ANTERIOR_12_MESES']) +  (df['INTENTO_SUICIDIO_12_MESES'])

    # Crear la columna 'CLASE' basado en 'INTENTO_SUICIDIO', 0=False y 1=True
    df['CLASE'] = (df['INTENTO_SUICIDIO'] != 0).astype(int)

    # Calcular el total de hijos
    df[['NUM_HIJOS', 'NUM_HIJAS']] = df[['NUM_HIJOS', 'NUM_HIJAS']].fillna(0)
    df['TOTAL_HIJOS'] = df['NUM_HIJOS'] + df['NUM_HIJAS']
   


    # 1Asegurar formato de fechas (reemplazamos '/' por '-')
    df[['ULT_ALTA_PPS', 'ULT_BAJA_PPS']] = df[['ULT_ALTA_PPS', 'ULT_BAJA_PPS']].astype(str).apply(lambda col: col.str.replace('/', '-'))

    # Convertir a datetime sin formato específico porque varía
    df[['ULT_ALTA_PPS_dt', 'ULT_BAJA_PPS_dt']] = df[['ULT_ALTA_PPS', 'ULT_BAJA_PPS']].apply(pd.to_datetime, errors='coerce')

    # Revisar fechas convertidas correctamente
    def detect_format(date_str):
        try:
            if pd.notna(date_str): 
                parsed_date = parser.parse(date_str)
                # Formato estándar
                return parsed_date.strftime('%Y-%m-%d')  
            return None
        except Exception:
            return None  

    def invalid_dates(df):
        for f, fd in zip(df['ULT_ALTA_PPS'], df['ULT_ALTA_PPS_dt']):
            if pd.notna(f) and pd.notna(fd):
                # Normalizar la fecha original
                original_format = detect_format(f)
                # Convertir `datetime64` al mismo formato 
                fd_str = fd.strftime('%Y-%m-%d')  
                #Comparamos las dos fechas
                if original_format and original_format != fd_str:
                    print(f'Original: {f} (Detectado: {original_format}), Datetime: {fd_str}')
              

    # Llamar a la función para verificar fechas convertidas
    invalid_dates(df)

    # Calcular el período entre fechas usando las columnas datetime
    df['PERIODO_ENTRE_PPS'] = (df['ULT_ALTA_PPS_dt'] - df['ULT_BAJA_PPS_dt']).dt.days.abs().fillna(0).astype(int)

    # Columnas queremos eliminar
    columnas_eliminar= ['NOMBRE_CENTRO','PROVINCIA_CENTRO', 'MUNICIPIO_CENTRO', 'FECHA_NACIMIENTO', 
                      'PROVINCIA_RESIDENCIA', 'NACIONALIDAD', 'SEXO', 'FECHA_ULTIMO_INGRESO', 
                      'FECHA_INGRESO_CENTRO_ACTUAL','FECHA_NACIMIENTO_HIJO_MAYOR', 'FECHA_NACIMIENTO_HIJO_MENOR',
                       'DELITOS', 'CONDENA_AAMMDD', 'FECHA_CUMPLIMIENTO_CONDENA', 'CT_APLICA', 'SIT_PENITENCIARIAS_ART75', 'TIPO_MODULO_UBICACION_INTERNO', 
                       'ACOMPANADO_CELDA', 'ACTIV_AREA_LABORAL_15_MESES', 'ACTIV_AREA_TERAPEUTICA_15_MESES','ACTIV_AREA_FORMATIVA_15_MESES', 
                       'ACTIV_AREA_OCUPACIONAL_15_MESES', 'ACTIV_AREA_DEPORTIVA_15_MESES', 
                       'ACTIV_AREA_EDUCATIVA_15_MESES', 'ACTIV_AREA_CULTURAL_15_MESES','NUM_ACTIV_AREA_CULTURAL_15_MESES',
                       'NUM_ACTIV_AREA_CULTURAL_3_MESES', 'NUM_ACTIV_AREA_EDUCATIVA_15_MESES',
                        'NUM_ACTIV_AREA_EDUCATIVA_3_MESES', 'NUM_ACTIV_AREA_EDUCATIVA_ACTUAL', 'NUM_ACTIV_AREA_DEPORTIVA_15_MESES',
                        'NUM_ACTIV_AREA_DEPORTIVA_3_MESES', 'NUM_ACTIV_AREA_OCUPACIONAL_15_MESES', 'NUM_ACTIV_AREA_OCUPACIONAL_3_MESES', 
                        'NUM_ACTIV_AREA_FORMATIVA_15_MESES', 'SITUACION_PENAL_INTERNO',
                        'NUM_ACTIV_AREA_FORMATIVA_3_MESES', 'NUM_ACTIV_AREA_TERAPEUTICA_15_MESES', 'NUM_ACTIV_AREA_TERAPEUTICA_3_MESES',
                        'NUM_ACTIV_AREA_LABORAL_15_MESES', 'NUM_ACTIV_AREA_LABORAL_3_MESES', 'PARENTESCO_COM_CONVIVENCIA_6_MESES', 'PARENTESCO_COM_FAMILIAR_6_MESES',
                        'PARENTESCO_COM_INTIMA_6_MESES', 'PARENTESCO_COM_LOCUTORIO_6_MESES', 'PARENTESCO_COM_VIDEOCONF_6_MESES',
                        'PARENTESCO_COM_VIDEOLLAM_6_MESES', 'PARENTESCO_COM_CITA_PROF_6_MESES', 'PARENTESCO_LOCUTORIOS_6_MESES',
                        'FALTAS_PENDIENTES_CUMPLIR', 'SANCIONES_POR_CUMPLIR', 'TIPO_AUTOLESIONES_ULT_12_MESES', 'NUM_HIJOS', 'NUM_HIJAS', 'AUTOLESIONES_ANTERIOR_12_MESES','AUTOLESIONES_ULT_12_MESES',
                        'INTENTO_SUICIDIO_ANTERIOR_12_MESES','TIPO_AUTOLESIONES_ANT_12_MESES', 
                        'TIPO_AUTOLESIONES_TOTAL', 'FECHA_SUCESO'
                        'INTENTO_SUICIDIO_12_MESES', 'ESTADO_CIVIL', 'DELITO_MAYOR_CONDENA', 'PARENTESCOS_INGRESOS', 'DELITO_MAYOR_CONDENA'
                        'ULT_ALTA_PPS', 'ULT_BAJA_PPS', 'ULT_BAJA_PPS_dt','ULT_ALTA_PPS_dt', 'FECHA_SUICIDO', 'Unnamed:_89', 'Unnamed:_90',  'FH_BAJA_CENTRO', 
                        'FECHA_DEL_INCIDENTE_(Suicidio)', 'INTENTO_SUCIDIO'
                        'FECHA_DEL_SUICIDIO', 'FECHA_SUCESO', 'FECHA_SUICIDIO','FECHA_DEL_SUICIDIO', 'INTENTO_SUICIDIO', 'INTENTO_SUICIDIO_12_MESES', 'ULT_ALTA_PPS', 'AÑO_NACIMIENTO'] # de momento ct_aplica y sit_penitenciaria los quitamos 
    
    # Check si estan (diferencias entre df)
    columnas_presentes = [col for col in columnas_eliminar if col in df.columns]

    # Eliminamos las presentes 
    df_dropped = df.drop(columns=columnas_presentes, axis=1)

    # Reset de los index para evitar problemas de alineación 
    df_dropped = df_dropped.reset_index(drop=True)
    encoded_nombre_centro = encoded_nombre_centro.reset_index(drop=True)
    encoded_provincia_residencia = encoded_provincia_residencia.reset_index(drop=True)
    encoded_nacionalidad = encoded_nacionalidad.reset_index(drop=True)
    encoded_sex = encoded_sex.reset_index(drop=True)
    encoded_estado_civil = encoded_estado_civil.reset_index(drop=True)
    situacion_penal = situacion_penal.reset_index(drop=True)
    tipo_modulo = tipo_modulo.reset_index(drop=True)
    acompañado_celda = acompañado_celda.reset_index(drop=True)
    delito_encoding = delito_encoding.reset_index(drop=True)

    df_final = pd.concat([df_dropped, 
                          encoded_nombre_centro, 
                          encoded_provincia_residencia, 
                          encoded_nacionalidad, 
                          encoded_sex, 
                          encoded_estado_civil, 
                          situacion_penal, tipo_modulo, acompañado_celda, delito_encoding
                          ], axis=1, ignore_index=False)

    return df_final
    


