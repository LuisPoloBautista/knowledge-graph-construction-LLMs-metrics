import json
import requests

# Función para buscar en Wikidata
def buscar_wikidata(entidad):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entidad}&language=es&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "search" in data and data["search"]:
            for result in data["search"]:
                if "id" in result:  # Recupera el ID si está presente
                    return f"https://www.wikidata.org/entity/{result['id']}"
    return None  # Si no se encuentra, devolver None

# Función para buscar con reducción progresiva de tokens
def buscar_con_reduccion(entidad):
    # Intentar buscar la entidad completa
    uri = buscar_wikidata(entidad)
    if uri:
        return uri

    # Reducir progresivamente la cantidad de tokens
    tokens = entidad.split()
    while len(tokens) > 1:
        tokens.pop()  # Eliminar el último token
        sub_entidad = " ".join(tokens)
        uri = buscar_wikidata(sub_entidad)
        if uri:
            return uri

    # Generar URI automáticamente si no se encuentra
    entidad_normalizada = entidad.replace(" ", "_")
    return f"https://example.org/entity/{entidad_normalizada}"

# Función para desambiguar y actualizar el JSON
def desambiguar_json(input_json):
    for elemento in input_json:
        for clave in ["head", "relation", "tail", "head_type", "tail_type"]:
            valor = elemento.get(clave, "")
            if valor:
                uri_clave = f"{clave}_uri"  # Crear el nombre de la clave URI
                if clave in ["head", "tail"]:
                    elemento[uri_clave] = buscar_con_reduccion(valor)  # Búsqueda con reducción de tokens
                else:
                    elemento[uri_clave] = buscar_wikidata(valor) or f"https://example.org/entity/{valor.replace(' ', '_')}"  # Búsqueda directa
    return input_json


def cargar_json():
    filename = "...c.json"
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data



# Descargar archivo JSON actualizado
def descargar_json(data, output_file="...json"):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    files.download(output_file)

# Flujo principal
if __name__ == "__main__":
    # Cargar el archivo JSON
    json_data = cargar_json()

    # Procesar el JSON para desambiguar entidades
    json_actualizado = desambiguar_json(json_data)

    # Descargar el archivo actualizado
    descargar_json(json_actualizado)