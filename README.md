<div align="center">
    <img src="img/KG-LLM-Metrics.png" alt="logo" style="border-radius: 50 px;">
</div>


<div align="center">
    <h1>Workflow for knowledge graph-construction using LLMs and metrics</h1>
</div>

This repository, KG-LLM-Metrics, provides a comprehensive workflow for building knowledge graphs (KGs) using large language models (LLMs) in a zero-shot approach. It includes methods for generating, evaluating, and optimizing knowledge graphs while leveraging a robust metrics framework that is designed to analyze the topological structure of the graph and assess the quality of the information relative to the original corpus, thereby achieving a more accurate and consistent representation of the extracted knowledge.

<h2 style="font-size: 2rem; margin-bottom: 20px;">Characteristics</h2>

The proposal is divided into nine sections: 

1) Corpus selection
2) Pre-processing
3) Topic modeling
4) Named entity and relations extraction using LLMs
5) Post-processing
6) Hallucination filtering
7) Semantic similarity
8) Word sense disambiguation
9) Knowledge graph creation

<h2 style="font-size: 2rem; margin-bottom: 20px;">Metrics</h2>

The metrics used were the following:

1. Number of nodes and edges
2. Percentage of overlap and differences
3. Frequency of specific types of entities
4. Clustering coefficient
5. Density
6. Average degree
7. Percentage of hallucination
8. Redundancy
9. Contextual relevance

<h2 style="font-size: 2rem; margin-bottom: 20px;">Large Language Models</h2>

We use the following LLMs for knowledge extraction. Scripts allow you to process data in a DataFrame, generating a new column with the results of the LLM model. Make sure you meet the dependencies before running it.

Requirements

Python 3.9 or higher.
Libraries: <a href="https://ollama.com/">ollama</a>, pandas, gc, NVIDIA GPU of at least 16GB recommended

<h2 style="font-size: 2rem; margin-bottom: 20px;">Llama 3.1</h2

Model Configuration: Edit the model in ollama.chat to adjust it to your needs. For example, you can replace llama3.1 with any other supported model.

```python
def process_text(text):
    response = ollama.chat(model='llama3.1', messages=[....])
    return response['message']['content']

def process_df(df, text_column, output_column):
    results = []
    for index, row in df.iterrows():
        text = row[text_column]
        result = process_text(text)
        results.append(result)

        del text
        del result
        gc.collect()

        print(f"Procesado fila {index + 1}/{len(df)}")

    df[output_column] = results
    return df
```


<h2 style="font-size: 2rem; margin-bottom: 20px;">Gemma 2</h2>

Similar to the case of Llama 3.1

```python
def process_text(text):
    response = ollama.chat(model='gemma2', messages=[....])
    return response['message']['content']

def process_df(df, text_column, output_column):
    results = []
    for index, row in df.iterrows():
        text = row[text_column]
        result = process_text(text)
        results.append(result)

        del text
        del result
        gc.collect()

        print(f"Procesado fila {index + 1}/{len(df)}")

    df[output_column] = results
    return df
```


