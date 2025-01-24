import pandas as pd

filename=""

# Lee el archivo JSON
with open(filename, 'r', encoding="latin9") as f:
    datos = pd.read_csv(f)

# Muestra el DataFrame
datos.head(3)


import re

def extract_brackets_content(text):
    # Handle non-string types by converting them to strings
    if not isinstance(text, str):
        text = str(text)
    # Utilizar una expresión regular para extraer el contenido entre corchetes
    matches = re.findall(r'\[.*?\]', text, re.DOTALL)
    return ' '.join(matches)

# Aplicar la función a la columna 'TEXT'
datos['Entidades GPT-4o'] = datos['Entidades GPT-4o'].apply(extract_brackets_content)


datos.head(3)

# Guardar el DataFrame procesado en un nuevo archivo JSON
output_filename = ''
datos.to_csv(output_filename, encoding="latin9")
print(f"Archivo procesado y guardado en {output_filename}")
