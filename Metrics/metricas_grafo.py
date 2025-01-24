import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

class MultiGrafoAnalizador:
    def __init__(self, archivos_json):
        """
        Inicializa el analizador con múltiples archivos JSON de tripletas.

        :param archivos_json: Lista de rutas a archivos JSON con tripletas.
        """
        self.grafos = []
        self.nombres_grafos = []
        self.metricas_por_grafo = []

        # Cargar y procesar cada archivo JSON
        for archivo in archivos_json:
            with open(archivo, 'r', encoding='utf-8') as f:
                tripletas = json.load(f)

            # Construir grafo
            G = self.construir_grafo(tripletas)
            self.grafos.append(G)
            self.nombres_grafos.append(archivo.split('/')[-1].split('.')[0])

    def construir_grafo(self, tripletas):
        """
        Construye un grafo de NetworkX a partir de las tripletas.

        :param tripletas: Lista de tripletas.
        :return: Grafo de NetworkX.
        """
        G = nx.DiGraph()

        # Añadir nodos y aristas
        for tripleta in tripletas:
            head = tripleta.get('head', '')
            tail = tripleta.get('tail', '')
            relation = tripleta.get('relation', '')

            if head and tail:  # Validar nodos
                G.add_node(head)
                G.add_node(tail)
                G.add_edge(head, tail, relation=relation)

        return G

    def calcular_metricas(self):
        """
        Calcula métricas avanzadas para todos los grafos.

        :return: Lista de diccionarios con métricas.
        """
        self.metricas_por_grafo = []

        for G, nombre in zip(self.grafos, self.nombres_grafos):
            # Métricas básicas
            metricas = {
                'nombre': nombre,
                'no_nodos': G.number_of_nodes(),
                'no_enlaces': G.number_of_edges(),
                'densidad': nx.density(G),
                'grado_medio': np.mean([d for n, d in G.degree()]),
                'coeficiente_agrupamiento': nx.average_clustering(G.to_undirected())
            }

            # PageRank
            pagerank = nx.pagerank(G)
            metricas['top_5_pagerank'] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]

            # Conectividad fuerte y débil
            try:
                componentes_fuertes = list(nx.strongly_connected_components(G))
                metricas['no_componentes_fuertes'] = len(componentes_fuertes)
                metricas['tamano_componente_fuerte_mas_grande'] = len(max(componentes_fuertes, key=len))

                componentes_debiles = list(nx.weakly_connected_components(G))
                metricas['no_componentes_debiles'] = len(componentes_debiles)
                metricas['tamano_componente_debil_mas_grande'] = len(max(componentes_debiles, key=len))
            except Exception as e:
                metricas['conectividad'] = f"Error al calcular: {str(e)}"

            # Centralidad
            centralidad_grado = nx.degree_centrality(G)
            metricas['top_5_centralidad_grado'] = sorted(centralidad_grado.items(), key=lambda x: x[1], reverse=True)[:5]

            self.metricas_por_grafo.append(metricas)

        return self.metricas_por_grafo

    def visualizar_comparacion_metricas(self):
        """
        Genera visualizaciones comparativas de métricas entre grafos.
        """
        df_metricas = pd.DataFrame(self.metricas_por_grafo)

        plt.figure(figsize=(20, 15))
        metricas_a_graficar = [
            ('no_nodos', 'Número de Nodos'),
            ('no_enlaces', 'Número de Enlaces'),
            ('densidad', 'Densidad del Grafo'),
            ('no_componentes_fuertes', 'Componentes Fuertemente Conectados'),
            ('no_componentes_debiles', 'Componentes Débilmente Conectados')
        ]

        for i, (metrica, titulo) in enumerate(metricas_a_graficar, 1):
            plt.subplot(3, 2, i)
            plt.bar(df_metricas['nombre'], df_metricas[metrica])
            plt.title(titulo)
            plt.xlabel('Grafo')
            plt.ylabel('Valor')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def exportar_metricas(self, nombre_archivo='metricas_grafos.json'):
        """
        Exporta las métricas de todos los grafos a un archivo JSON.

        :param nombre_archivo: Nombre del archivo de salida.
        """
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            json.dump(self.metricas_por_grafo, f, indent=2, ensure_ascii=False)

        print(f"Métricas exportadas a {nombre_archivo}")

# Ejemplo de uso
if __name__ == '__main__':
    # Especificar las rutas a los archivos JSON
    archivos_json = [
        'tripletas_gemma_.json',
        'tripletas_llama.json',
        'tripletas_olmo.json',
        'tripletas_gpt.json'
    ]

    # Crear analizador
    analizador = MultiGrafoAnalizador(archivos_json)

    # Calcular métricas
    metricas = analizador.calcular_metricas()

    # Imprimir métricas de cada grafo
    print("\nMétricas de Grafos:")
    for grafo_metricas in metricas:
        print(f"\nGrafo: {grafo_metricas['nombre']}")
        for metrica, valor in grafo_metricas.items():
            print(f"{metrica}: {valor}")

    # Visualizar comparación de métricas
    analizador.visualizar_comparacion_metricas()

    # Exportar métricas
    analizador.exportar_metricas()
