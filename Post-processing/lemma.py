import json
import spacy
import os


nlp=spacy.load("es_core_news_lg")

filename = "....json"

# Carga el archivo JSON de entrada
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Constantes para las claves del JSON
HEAD = 'head'
HEAD_TYPE = 'head_type'
#RELATION = 'relation'  # Comentar/Descomentar según sea necesario
TAIL = 'tail'
TAIL_TYPE = 'tail_type'

# Función para lematizar un texto
def lemmatize_text(text):
    if text is not None:
        if isinstance(text, list):
            return ' '.join([lemmatize_text(item) for item in text if item])
        elif isinstance(text, str):  # Verificar que text sea de tipo string
            doc = nlp(text)
            return ' '.join(token.lemma_ for token in doc)
    return ''  # Retorna una cadena vacía si text es None o no es válido


# Procesar y lematizar cada entrada en los datos
for entry in data:
    if isinstance(entry, dict):
        entry[HEAD] = lemmatize_text(entry.get(HEAD, ''))
        entry[HEAD_TYPE] = lemmatize_text(entry.get(HEAD_TYPE, ''))
        #entry[RELATION] = lemmatize_text(entry.get(RELATION, ''))  # Descomentar si es necesario
        entry[TAIL] = lemmatize_text(entry.get(TAIL, ''))
        entry[TAIL_TYPE] = lemmatize_text(entry.get(TAIL_TYPE, ''))

# Guardar el archivo JSON actualizado
output_filename = '....json'
with open(output_filename, 'w',  encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Archivo actualizado guardado como {output_filename}.")

