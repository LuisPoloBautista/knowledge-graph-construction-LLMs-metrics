import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
#from google.colab import files
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from collections import Counter


filename = "....json"

# Cargar los datos JSON
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Cargar el modelo de embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Obtener todos los "head_type", verificando que cada entrada sea un diccionario
head_types = [entry.get('head_type') for entry in data if isinstance(entry, dict) and entry.get('head_type')]

# Vectorizar los "head_type"
head_type_embeddings = model.encode(head_types)

# Visualización de los embeddings originales en el espacio vectorial
def visualizar_embeddings(embeddings, labels, title, zoom=False, zoom_threshold=5):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Crear la figura más ancha
    plt.figure(figsize=(14, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='orange')

    # Ajustar el tamaño de las etiquetas para que sean más pequeñas
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=6)

    plt.title(title)

    # Si se activa el zoom, se hará un acercamiento en áreas densas
    if zoom:
        # Calcular distancias entre los puntos
        distances = pdist(reduced_embeddings)
        distance_matrix = squareform(distances)

        # Fijar un umbral para definir "densidad"
        close_pairs = np.where((distance_matrix > 0) & (distance_matrix < np.percentile(distances, zoom_threshold)))

        if close_pairs[0].size > 0:  # Asegurar que se encontraron pares cercanos
            # Crear una nueva figura con el acercamiento
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='orange')

            for i, label in enumerate(labels):
                plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=6)

            # Dibujar un rectángulo que delimite la región densa
            x_min = reduced_embeddings[close_pairs[0], 0].min()
            x_max = reduced_embeddings[close_pairs[0], 0].max()
            y_min = reduced_embeddings[close_pairs[1], 1].min()
            y_max = reduced_embeddings[close_pairs[1], 1].max()

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title(f'Zoom en áreas densas - {title}')
            plt.show()
        else:
            print("No se encontraron nodos cercanos para hacer zoom.")

    # Mostrar la figura principal
    plt.show()

# Visualización antes de la unificación
visualizar_embeddings(head_type_embeddings, head_types, 'Embeddings de tail (Antes de Unificar)', zoom=True, zoom_threshold=5)

# Umbral para considerar que dos palabras son similares (ajustar según sea necesario)
similarity_threshold = 0.7

# Unificar palabras semánticamente similares
unified_map = {}
for i, head_type in enumerate(head_types):
    if head_type not in unified_map:
        # Comparar la similitud de la palabra actual con las demás
        similarities = cosine_similarity([head_type_embeddings[i]], head_type_embeddings)[0]
        similar_words = [head_types[j] for j in range(len(similarities)) if similarities[j] >= similarity_threshold]

        # Escoger la palabra de referencia (la primera que cumple la condición)
        unified_map[head_type] = similar_words[0]

# Reemplazar los "head_type" unificados en el JSON
for entry in data:
    if isinstance(entry, dict) and entry.get('head_type'):  # Verificar que 'head_type' exista y que sea un diccionario
        entry['head_type'] = unified_map[entry['head_type']]

# Guardar el JSON modificado
with open('....json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# Obtener los "head_type" unificados
unified_head_types = [unified_map[ht] for ht in head_types]

# Vectorizar los "head_type" unificados
unified_head_type_embeddings = model.encode(unified_head_types)

# Visualización después de la unificación
visualizar_embeddings(unified_head_type_embeddings, unified_head_types, 'Embeddings de head_type (Después de Unificar)', zoom=True, zoom_threshold=5)

# Mostrar las palabras que se unificaron
print("Palabras unificadas:")
for original, unified in unified_map.items():
    if original != unified:
        print(f"{original} -> {unified}")
