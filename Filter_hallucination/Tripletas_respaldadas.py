import pandas as pd
import json

# Función para alinear oraciones con tripletas y seleccionar las que tienen resultado 0
def obtener_tripletas_no_respaldo(row):
    try:
        # Cargar las tripletas como JSON
        tripletas = json.loads(row['Entidades GPT-4o'])
        # Cargar los detalles de las oraciones de la columna 'detalles_tripletas' como JSON
        detalles = json.loads(row['detalles_tripletas'])
        
        # Lista para almacenar las tripletas que representan oraciones no respaldadas
        tripletas_no_respaldo = []

        # Iterar sobre los detalles para identificar las oraciones con resultado 0
        for detalle in detalles:
            if detalle['resultado'] == 0:
                # Buscar la tripleta correspondiente que generó esta oración
                for tripleta in tripletas:
                    oracion_generada = f"{tripleta['head']} {tripleta['relation']} {tripleta['tail']}."
                    if oracion_generada == detalle['oracion']:
                        tripletas_no_respaldo.append(tripleta)
                        break
        
        # Devolver la lista de tripletas no respaldadas como JSON
        return json.dumps(tripletas_no_respaldo, ensure_ascii=False)
    
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # En caso de error, devolver un valor vacío
        print(f"Error al procesar la fila: {e}")
        return json.dumps([], ensure_ascii=False)

# Cargar el archivo CSV
filename = ''
df = pd.read_csv(filename, encoding="latin9")

# Aplicar la función a cada fila del DataFrame para crear la nueva columna
df['tripletas_respaldadas'] = df.apply(obtener_tripletas_no_respaldo, axis=1)

# Guardar el nuevo DataFrame en un archivo CSV
df.to_csv('', index=False, encoding="latin9")

print("Script completado")