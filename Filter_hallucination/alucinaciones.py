import pandas as pd
import gc
import ollama
import json
from tqdm import tqdm
import time

# Función para convertir una tripleta en una oración o afirmación legible
def convertir_tripleta_a_oracion(tripleta):
    try:
        head = str(tripleta.get('head', ''))
        relation = str(tripleta.get('relation', ''))
        tail = str(tripleta.get('tail', ''))

        # Si alguna de las claves esenciales no está presente, omitir la tripleta
        if not head or not relation or not tail:
            raise KeyError("Tripleta incompleta")
        
        return f"{head} {relation} {tail}."
    except KeyError as e:
        # Imprimir un mensaje de depuración y continuar con la siguiente tripleta
        print(f"Tripleta ignorada: {e}")
        return ""
    except Exception as e:
        print(f"Error al procesar la tripleta: {e}")
        return ""

# Función para verificar si una afirmación está respaldada por la noticia
def verificar_alucinacion(noticia, oracion):
    try:
        prompt = f"Document:\n{noticia}\n\nAfirmación:\n{oracion}\n\n¿La afirmación está respaldada por el documento? Responde con Sí o No."
        # Llamada al modelo bespoke-minicheck
        respuesta = ollama.chat(model="bespoke-minicheck", messages=[{"role": "system", "content": prompt}])
        return 1 if 'No' in respuesta['message']['content'] else 0
    except Exception as e:
        print(f"Error al verificar la afirmación: {e}")
        return 0

# Función para procesar cada fila y calcular el porcentaje de alucinación por tripleta
def procesar_fila(fila):
    noticia = fila['texto_completo']  # Asumiendo que el documento está en una columna llamada 'texto_completo'
    json_afirmacion = str(fila['Entidades GPT-4o'])  # Aseguramos que el valor sea tratado como string

    try:
        afirmaciones_json = json.loads(json_afirmacion)
    except json.JSONDecodeError:
        print("JSON inválido, omitiendo fila.")
        return pd.Series([0, json.dumps([], ensure_ascii=False), 0])  # Si no se puede cargar el JSON, retornamos 0 y una lista vacía

    total_tripletas = len(afirmaciones_json)
    if total_tripletas == 0:
        return pd.Series([0, json.dumps([], ensure_ascii=False), 0])  # No hay tripletas para procesar

    alucinaciones_tripletas = 0
    detalles_tripletas = []  # Para almacenar los resultados de cada tripleta

    for tripleta in afirmaciones_json:
        oracion = convertir_tripleta_a_oracion(tripleta)
        if oracion:  # Verificamos solo si la oración no está vacía
            try:
                resultado_alucinacion = verificar_alucinacion(noticia, oracion)
                detalles_tripletas.append({'oracion': oracion, 'resultado': resultado_alucinacion})
                alucinaciones_tripletas += resultado_alucinacion
            except Exception as e:
                print(f"Error al procesar la oración '{oracion}': {e}")
                continue  # Pasar a la siguiente tripleta en caso de error

    # Calculamos el porcentaje de alucinación por tripleta
    porcentaje_alucinacion = (alucinaciones_tripletas / total_tripletas) * 100
    return pd.Series([porcentaje_alucinacion, json.dumps(detalles_tripletas, ensure_ascii=False), total_tripletas])  # Retornamos los detalles y el número de tripletas

# Cargar archivo localmente (modificar el nombre de archivo si es necesario)
filename = ''
data = pd.read_csv(filename, encoding="latin9")

# Seleccionar un subconjunto de datos para pruebas si es necesario
#df = data.iloc[0:942]

# Usar tqdm para mostrar el progreso del procesamiento
start_time = time.time()
tqdm.pandas(desc="Procesando filas")

# Aplicar la función con progreso
data[['porcentaje_alucinacion', 'detalles_tripletas', 'total_tripletas']] = data.progress_apply(procesar_fila, axis=1)

# Calcular métricas generales
promedio_porcentaje_alucinacion = data['porcentaje_alucinacion'].mean()
total_tripletas_evaluadas = data['total_tripletas'].sum()
total_documentos = len(data)

# Imprimir métricas finales
print(f"\nMétricas generales:")
print(f"Porcentaje promedio de alucinación en todos los documentos: {promedio_porcentaje_alucinacion:.2f}%")
print(f"Total de tripletas evaluadas: {total_tripletas_evaluadas}")
print(f"Total de documentos procesados: {total_documentos}")

# Guardar el nuevo CSV con la columna de porcentaje de alucinación y detalles de tripletas
data.to_csv('', index=False, encoding="latin9")

end_time = time.time()
elapsed_time = (end_time - start_time) / 60  # Tiempo en minutos
print(f"\nArchivo procesado y guardado. Tiempo estimado: {elapsed_time:.2f} minutos")
