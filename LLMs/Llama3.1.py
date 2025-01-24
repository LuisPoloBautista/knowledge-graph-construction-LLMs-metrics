import pandas as pd
import gc
import ollama

# Cargar archivo localmente (modificar el nombre de archivo si es necesario)
filename = ''
data = pd.read_csv(filename, encoding="latin8")

data.info()

# --- Uso de Ollama localmente para extraer tripletas ---
# Asegúrate de que el servidor de Ollama esté ejecutándose localmente.

# Función para procesar texto con el modelo de Ollama
def process_text(text):
    response = ollama.chat(model='llama3.1', messages=[
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

            **Ejemplo de formato JSON (es solo es un ejemplo, no lo copies ni uses en producción)**:

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

# Función para procesar todo el DataFrame
def process_df(df, text_column, output_column):
    results = []
    for index, row in df.iterrows():
        text = row[text_column]
        result = process_text(text)
        results.append(result)

        # Liberar memoria
        del text
        del result
        gc.collect()

        # Imprimir progreso
        print(f"Procesado fila {index + 1}/{len(df)}")

    df[output_column] = results
    return df

# Procesar el DataFrame
text_column = 'texto_completo'  # Cambia esto según el nombre de tu columna de texto
output_column = 'TripletasLlama'  # Cambia esto según el nombre de la columna de salida
df_tripletas = process_df(df, text_column, output_column)

# Guardar el DataFrame procesado en un nuevo archivo CSV
output_filename = ''

df_tripletas.to_csv(output_filename, index=False)
print(f"Archivo procesado y guardado en {output_filename}")
