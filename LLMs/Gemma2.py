import pandas as pd
import gc
import ollama

# Cargar archivo localmente (modificar el nombre de archivo si es necesario)
filename = ''

# Crear un iterador para leer el archivo en chunks
chunk_size = 1000  # Tamaño del chunk
df_iterator = pd.read_csv(filename, encoding="latin9", chunksize=chunk_size)

# --- Uso de Ollama localmente para extraer tripletas ---
# Asegúrate de que el servidor de Ollama esté ejecutándose localmente.

# Función para procesar texto con el modelo de Ollama
def process_text(text):
    response = ollama.chat(model='gemma2', messages=[
        {
            'role': 'user',
            'content': f"""A partir del siguiente texto relacionado con desastres naturales: {text},
            realiza las siguientes tareas:

            1. **Extraer entidades nombradas**: Identifica todas las entidades relevantes en el texto.
            2. **Identificar relaciones**: Establece las relaciones entre las entidades extraídas.

            **Requisitos de Salida**:
            - Formato: Debe devolver un objeto JSON.
            - Estructura del objeto: Cada entidad y su relación deben estar representadas con las siguientes claves:
                - **head**: Entidad principal.
                - **head_type**: Tipo de la entidad principal (e.g., 'persona', 'lugar', 'fecha').
                - **relation**: Relación entre la entidad principal y la secundaria (e.g., 'ubicado en', 'causado por').
                - **tail**: Entidad secundaria.
                - **tail_type**: Tipo de la entidad secundaria (e.g., 'organización', 'infraestructura').

            **Ejemplo de formato JSON (es solo es un ejemplo, no lo copies ni lo uses en producción)**:

            [
                {{
                    "head": "Entidad principal",
                    "head_type": "Tipo de la entidad principal",
                    "relation": "Relación entre la entidad principal y la secundaria",
                    "tail": "Entidad secundaria",
                    "tail_type": "Tipo de la entidad secundaria"
                }},
                {{
                    "head": "Entidad principal",
                    "head_type": "Tipo de la entidad principal",
                    "relation": "Relación entre la entidad principal y la secundaria",
                    "tail": "Entidad secundaria",
                    "tail_type": "Tipo de la entidad secundaria"
                }}
            ]
            """
        },
    ])
    return response['message']['content']

# Procesar cada chunk
output_filename = 'procesado_gemma.csv'
first_chunk = True  # Bandera para indicar si es el primer chunk

for chunk_number, chunk in enumerate(df_iterator, start=1):
    print(f"Procesando chunk {chunk_number}...")

    # Procesar cada fila del chunk
    chunk['TripletasGemma'] = chunk['texto_completo'].apply(process_text)

    # Guardar resultados en el archivo CSV
    if first_chunk:
        chunk.to_csv(output_filename, index=False, mode='w')  # Escribir encabezados
        first_chunk = False
    else:
        chunk.to_csv(output_filename, index=False, mode='a', header=False)  # Añadir sin encabezados

    # Liberar memoria
    del chunk
    gc.collect()

    print(f"Chunk {chunk_number} procesado y guardado.")

print(f"Todos los chunks han sido procesados y guardados en {output_filename}")

