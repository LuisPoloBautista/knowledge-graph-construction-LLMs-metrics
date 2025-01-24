
from google.colab import files
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Obtener el nombre del archivo
filename = ""


# Leer el archivo CSV
datos = pd.read_csv(filename, encoding='UTF-8')

# Mostrar el DataFrame
datos.head(3)

# Asegurarse de tener los stopwords de NLTK
nltk.download('stopwords')
stop_words = stopwords.words('spanish')
nltk.download('punkt')

def preprocess(text):

    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens
    else:
        return []

# Preprocesar el texto, asegurándose de que 'texto_completo' sea de tipo texto
datos['processed_text'] = datos['texto_completo'].astype(str).apply(preprocess)

!pip install spacy
!python -m spacy download es_core_news_sm

import spacy

nlp = spacy.load("es_core_news_sm")

# Función para lematizar una lista de tokens
def lemmatize(tokens):
    text = ' '.join(tokens)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return lemmas

datos['processed_text'] = datos['processed_text'].apply(lemmatize)

datos.head(3)

from gensim.models import LdaModel
import random
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from gensim.corpora import Dictionary

diccionario = Dictionary(datos.processed_text)
print(f'Número de tokens: {len(diccionario)}')

diccionario.filter_extremes(no_below=2, no_above = 0.8)
print(f'Número de tokens: {len(diccionario)}')

corpus = [diccionario.doc2bow(tweet) for tweet in datos.processed_text]

print(corpus[6])

lda = LdaModel(corpus=corpus, id2word=diccionario, num_topics=20, random_state=42,
               chunksize=1000, passes=100, alpha='auto')

topicos = lda.print_topics(num_words=5, num_topics=20)
for topico in topicos:
    print(topico)

!pip install pyLDAvis

import pyLDAvis.gensim_models
from google.colab import files

# Suponiendo que ya tienes el modelo LDA, corpus y diccionario preparados
# Preparar la visualización
vis = pyLDAvis.gensim_models.prepare(lda, corpus, diccionario)

# Guardar la visualización en un archivo HTML
archivo_html = '...html'
pyLDAvis.save_html(vis, archivo_html)

# Descargar el archivo a tu computadora local
files.download(archivo_html)

for i in range(0, 10):
    plt.figure()
    plt.imshow(WordCloud(background_color='white', prefer_horizontal=1.0).fit_words(dict(lda.show_topic(i, 20))))
    plt.axis("off")
    plt.title("Tópico " + str(i))
    plt.show()

indice_noticia = random.randint(0,len(datos))
noticia = datos.iloc[indice_noticia]
print(noticia.processed_text)

bow_noticia = corpus[indice_noticia]
distribucion_noticia = lda[bow_noticia]

dist_indices = [topico[0] for topico in lda[bow_noticia]]
# Contribución de los topicos mas significativos
dist_contrib = [topico[1] for topico in lda[bow_noticia]]

distribucion_topicos = pd.DataFrame({'Topico':dist_indices,
                                     'Contribucion':dist_contrib })
distribucion_topicos.sort_values('Contribucion',
                                 ascending=False, inplace=True)
ax = distribucion_topicos.plot.bar(y='Contribucion',x='Topico',
                                   rot=0, color="orange",
                                   title = 'Tópicos mas importantes'
                                   'de noticia ' + str(indice_noticia))

for ind, topico in distribucion_topicos.iterrows():
    print("*** Tópico: " + str(int(topico.Topico)) + " ***")
    palabras = [palabra[0] for palabra in lda.show_topic(
        topicid=int(topico.Topico))]
    palabras = ', '.join(palabras)
    print(palabras, "\n")

datos['topico']=''

for i in range(0,len(datos)):
    print(i)
    indice_noticia = i
    noticia = datos.iloc[indice_noticia]
    bow_noticia = corpus[indice_noticia]
    distribucion_noticia = lda[bow_noticia]
    dist_indices = [topico[0] for topico in lda[bow_noticia]]
    # Contribución de los topicos mas significativos
    dist_contrib = [topico[1] for topico in lda[bow_noticia]]
    # Ensure both lists have the same length - filling with None if necessary
    max_length = max(len(dist_indices), len(dist_contrib))
    dist_indices += [None] * (max_length - len(dist_indices))
    dist_contrib += [None] * (max_length - len(dist_contrib))
    distribucion_topicos = pd.DataFrame({'Topico':dist_indices,
                                     'Contribucion':dist_contrib })
    distribucion_topicos.sort_values('Contribucion',
                                 ascending=False, inplace=True)
    datos['topico'].iloc[i]=distribucion_topicos.Topico.iloc[0]

datos.head()

output_filename = '....csv'
datos.to_csv(output_filename)
print(f"Archivo procesado y guardado en {output_filename}")

files.download(output_filename)