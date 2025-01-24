from openai import OpenAI
import pandas as pd
import os
import json
import csv

# Configura tu clave de API de OpenAI
client = OpenAI(
    api_key="...")

# Nombre del archivo CSV de entrada
input_csv = ''

output_csv = ''


def obtener_tripletas(texto):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"""A partir del siguiente texto relacionado con desastres naturales: {texto},
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
        """,}],
        model="gpt-4o",
        #messages=[{"role": "user", "content": prompt}],
        max_tokens=750,
        n=1,
        stop=None,
        temperature=0.5,
    )

    tripletas_json = response.choices[0].message.content
    return tripletas_json

datos = pd.read_csv(input_csv, encoding='latin9')


df['Entidades GPT-4o'] = df['texto_completo'].apply(obtener_tripletas) #Nombre de la columna

df.to_csv(output_csv, index=False, encoding='latin9')

print("Proceso completado.")