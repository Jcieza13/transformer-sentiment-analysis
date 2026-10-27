# 🎬 Transformer Sentiment Analysis

Proyecto académico realizado como parte del programa Nanodegree  **Future AWS AI Programmer de Udacity**.

Implementación y entrenamiento de una arquitectura Transformer para realizar análisis de sentimiento sobre reseñas de películas utilizando el conjunto de datos **IMDB Movie Reviews Dataset**.

El objetivo es clasificar automáticamente las reseñas como positivas o negativas mediante técnicas de Procesamiento de Lenguaje Natural (*Natural Language Processing*, NLP) y *Deep Learning*.

---

## 📋 Descripción

Este proyecto consiste en implementar y entrenar una arquitectura Transformer para realizar clasificación binaria de sentimientos.

La estructura base del proyecto fue proporcionada por Udacity. El trabajo realizado consistió en completar la implementación del pipeline de procesamiento de datos, adaptar la arquitectura Transformer para clasificación binaria, entrenar el modelo y evaluar su rendimiento sobre datos no vistos.

El desarrollo abarcó desde la preparación y tokenización del texto hasta la implementación de los mecanismos de atención, el entrenamiento de la red neuronal y la evaluación de los resultados.

## 🎯 Objetivos

Los principales objetivos del proyecto fueron:

- Cargar y explorar el conjunto de datos IMDB.
- Preparar los datos para tareas de Procesamiento de Lenguaje Natural.
- Implementar tokenización basada en BERT.
- Construir un `Dataset` personalizado utilizando PyTorch.
- Crear `DataLoaders` para entrenamiento, validación y prueba.
- Adaptar una arquitectura Transformer para clasificación binaria.
- Entrenar y evaluar el modelo.
- Analizar el rendimiento mediante métricas de *accuracy*.

---

## 🛠️ Tecnologías y herramientas utilizadas

- **Python:** lenguaje de programación.
- **PyTorch:** implementación y entrenamiento de la red neuronal.
- **BERT Tokenizer:** tokenización del texto.
- **Transformers:** arquitectura basada en mecanismos de atención.
- **NLP:** procesamiento de lenguaje natural.
- **Deep Learning:** entrenamiento de modelos de aprendizaje profundo.
- **IMDB Movie Reviews Dataset:** conjunto de datos para clasificación de sentimientos.

---

## 📊 Dataset

Se utilizó el conjunto de datos **IMDB Movie Reviews Dataset**, compuesto por 50.000 reseñas de películas distribuidas en dos conjuntos.

| Conjunto | Cantidad de reseñas |
|:---|---:|
| Entrenamiento | 25.000 |
| Prueba | 25.000 |
| **Total** | **50.000** |

Las reseñas están etiquetadas según su sentimiento:

| Etiqueta | Sentimiento |
|---:|---|
| `1` | Positivo |
| `0` | Negativo |

El objetivo es que el modelo aprenda patrones lingüísticos asociados a cada categoría y pueda clasificar correctamente reseñas que no haya visto durante el entrenamiento.

---

## 🔤 Preprocesamiento de texto

Para la tokenización se utilizó el tokenizador preentrenado:

```python
bert-base-uncased
```

Este tokenizador permite transformar el texto en secuencias de tokens que pueden procesarse mediante una arquitectura de Deep Learning.

### Características implementadas

- **Subword Tokenization:** división del texto en unidades léxicas que pueden incluir subpalabras.
- **Padding:** igualación de la longitud de las secuencias dentro de un lote.
- **Truncation:** limitación de las secuencias que superan la longitud máxima.
- **Longitud máxima:** 128 tokens por reseña.

Estas operaciones permiten preparar las reseñas para su procesamiento eficiente durante las etapas de entrenamiento, validación y prueba.

---

## 🏗️ Arquitectura implementada

El proyecto incluye la implementación de los componentes principales de una arquitectura Transformer.

### Componentes principales

- `AttentionHead`
- `MultiHeadAttention`
- `FeedForward Network`
- `Transformer Block`
- `DemoGPT`, adaptado para clasificación binaria

### Componentes adicionales

Para adaptar la arquitectura a la tarea de análisis de sentimiento, se incorporaron los siguientes elementos:

- **Mean Pooling:** agregación de las representaciones de los tokens.
- **Classification Head:** capa final para predecir el sentimiento.
- **Cross Entropy Loss:** función de pérdida para clasificación multiclase, utilizada aquí con dos categorías.
- **AdamW Optimizer:** optimizador para actualizar los parámetros del modelo durante el entrenamiento.

La arquitectura procesa las secuencias de tokens, genera representaciones contextuales del texto y utiliza una capa de clasificación para predecir si cada reseña expresa un sentimiento positivo o negativo.

---

## ⚙️ Configuración del modelo

Los principales hiperparámetros utilizados fueron los siguientes:

| Parámetro | Valor |
|:---|---:|
| Embedding Dimension | 128 |
| Transformer Layers | 4 |
| Attention Heads | 4 |
| Head Size | 32 |
| Batch Size | 32 |
| Epochs | 6 |
| Learning Rate | `3e-4` |
| Maximum Sequence Length | 128 |

Esta configuración define la capacidad de representación del modelo, la cantidad de bloques Transformer, el tamaño de los lotes y los parámetros principales del proceso de optimización.

---

## 🚀 Entrenamiento

Durante el entrenamiento se implementaron las siguientes etapas:

1. **Forward Pass:** procesamiento de las secuencias de entrada para obtener las predicciones.
2. **Cálculo de la pérdida:** utilización de `Cross Entropy Loss` para medir la diferencia entre las predicciones y las etiquetas reales.
3. **Backpropagation:** cálculo de gradientes para actualizar los parámetros de la red neuronal.
4. **Optimización:** actualización de los parámetros mediante `AdamW`.
5. **Validación:** evaluación del modelo al final de cada época para analizar su rendimiento sobre datos no utilizados directamente para actualizar sus parámetros.

### 📈 Evolución de la precisión de validación

La siguiente tabla muestra la evolución de la *validation accuracy* durante las seis épocas de entrenamiento.

| Época | Validation Accuracy |
|---:|---:|
| 1 | 72.80% |
| 2 | 76.08% |
| 3 | 77.92% |
| 4 | **79.24%** |
| 5 | 77.80% |
| 6 | 77.96% |

El mejor resultado se obtuvo durante la **cuarta época**, con una precisión de validación del **79.24%**.

---

# 📊 Resultados

## Resultados generales

El modelo fue evaluado utilizando los conjuntos de validación y prueba.

### Accuracy de validación

| Métrica | Resultado |
|:---|---:|
| Mejor Validation Accuracy | **79.24%** |

### Accuracy de prueba

| Métrica | Resultado |
|:---|---:|
| Test Accuracy | **76.26%** |

### ✅ Cumplimiento del objetivo

El modelo alcanzó una precisión del **76.26% sobre el conjunto de prueba**, superando el requisito del proyecto, que exigía una precisión superior al 75%.

Esto representa un resultado satisfactorio para la implementación desarrollada y demuestra que el modelo aprendió patrones relevantes para distinguir entre reseñas positivas y negativas.

---

## 🔍 Análisis de resultados

### Evolución durante el entrenamiento

El modelo mostró una mejora progresiva durante las primeras cuatro épocas, aumentando su precisión de validación desde el **72.80%** hasta el **79.24%**.

Después de la cuarta época, la precisión disminuyó ligeramente, situándose en un 77.80% durante la quinta época y recuperándose hasta un 77.96% en la sexta.

Este comportamiento indica que la mayor mejora se produjo durante las primeras etapas del entrenamiento y que las épocas posteriores no superaron el mejor resultado de validación obtenido.

### Capacidad de generalización

La precisión final sobre el conjunto de prueba fue del **76.26%**, inferior al máximo del 79.24% alcanzado en validación.

La diferencia entre ambos resultados sugiere que el rendimiento del modelo varía entre los conjuntos evaluados. Aun así, el modelo superó el umbral de precisión establecido para el proyecto y mostró capacidad para clasificar reseñas no vistas durante el entrenamiento.

### Interpretación general

Los resultados demuestran que la arquitectura Transformer implementada fue capaz de aprender patrones lingüísticos relacionados con el sentimiento expresado en las reseñas de películas.

Sin embargo, existe margen para mejorar su rendimiento mediante ajustes en los hiperparámetros, estrategias de regularización y experimentos con distintas configuraciones de la arquitectura.

---

## 🧠 Aprendizajes

Durante el desarrollo del proyecto se trabajó con los siguientes conceptos y herramientas:

- Procesamiento de Lenguaje Natural (*Natural Language Processing*, NLP).
- Tokenización basada en BERT.
- Construcción de `Dataset` personalizados en PyTorch.
- Uso de `DataLoaders` para gestionar los datos.
- Implementación de arquitecturas Transformer.
- Mecanismos de atención (*Attention Mechanisms*).
- Clasificación binaria de texto.
- Funciones de pérdida para clasificación.
- Optimización mediante AdamW.
- Entrenamiento y validación de modelos de Deep Learning.
- Evaluación mediante métricas de *accuracy*.
- Análisis de resultados y capacidad de generalización.

---

# 🏆 Conclusión

Se implementó exitosamente una arquitectura Transformer para clasificación binaria de sentimientos utilizando el conjunto de datos IMDB Movie Reviews Dataset.

El modelo alcanzó una precisión de prueba del **76.26%**, superando el objetivo establecido en el proyecto y demostrando su capacidad para distinguir entre reseñas positivas y negativas.

Entre los principales resultados destacan:

- **79.24%** de precisión máxima sobre el conjunto de validación.
- **76.26%** de precisión sobre el conjunto de prueba.
- Implementación de los principales componentes de una arquitectura Transformer.
- Integración de tokenización basada en BERT con un pipeline de entrenamiento en PyTorch.

Además de obtener resultados satisfactorios, el proyecto permitió profundizar en conceptos fundamentales de NLP, tokenización, mecanismos de atención y entrenamiento de modelos Transformer.

En conjunto, este trabajo constituye una experiencia práctica en la implementación y evaluación de arquitecturas de Deep Learning aplicadas al análisis de sentimiento.

---

## 👨‍💻 Autor

**Jen Pierr Ismael Cieza Diaz**

Proyecto académico realizado como parte del programa **AWS AI Programmer de Udacity**.
