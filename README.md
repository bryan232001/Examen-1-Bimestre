# Examen Primer Bimestre — Recuperación de Información (IR26A)
**Estudiante:** Bryan Yunga  
**Institución:** Escuela Politécnica Nacional (EPN) — FIS  
**Entorno de ejecución sugerido:** Kaggle Notebooks (Environment Avanzado)  

---

## 📋 Descripción General del Proyecto
Este proyecto implementa un pipeline completo de Procesamiento de Lenguaje Natural (NLP) enfocado en la **generación, búsqueda y visualización de embeddings semánticos** a partir del dataset público **Rotten Tomatoes Movies and Critic Reviews** obtenido de Kaggle. 

El motor central utiliza arquitecturas transformadoras ligeras para mapear texto libre en un espacio vectorial continuo de alta dimensionalidad. Posteriormente, permite realizar búsquedas por similitud de coseno frente a queries en lenguaje natural, y analiza la topología de dicho espacio reduciendo sus componentes principales a un plano bidimensional.

---

## 🛠️ Arquitectura del Pipeline y Celdas de Ejecución

### Celda 1: Instalación de Dependencias Críticas
Configura el entorno de ejecución asegurando la compatibilidad de versiones específicas y suprimiendo logs ruidosos o alertas visuales innecesarias del gestor de paquetes (`pip`).
* **Comando clave:** `!pip install -q --disable-pip-version-check ...`
* **Librerías principales:** `sentence-transformers==3.0.1`, `scikit-learn==1.5.1`, `pandas==2.2.2`, `numpy==1.26.4`, `matplotlib==3.9.1`.

### Celda 2: Importaciones y Configuración Global
Importa los módulos de manipulación de datos, NLP y visualización gráfica. Establece un estado de aleatoriedad fijo (`RANDOM_SEED = 42`) para garantizar la total reproducibilidad de los muestreos, entrenamientos y proyecciones de datos. Asimismo, descarga los recursos léxicos de `NLTK` (`stopwords`, `wordnet`).

### Celda 3: Carga del Corpus desde Kaggle
Automatiza la ingesta de datos leyendo directamente desde las rutas relativas provistas por la API de Kaggle. Carga las estructuras tabulares base:
1.  `rotten_tomatoes_movies.csv`: Contiene la metadata de las películas (17,712 registros).
2.  `rotten_tomatoes_critic_reviews.csv`: Contiene las reseñas textuales de los críticos de cine (1,130,017 registros).

### Celda 4: Selección de Campos y Construcción de Documentos
Aplica una estrategia de consolidación semántica uniendo (*joining*) las tablas por su identificador único (`rotten_tomatoes_link`). Construye cada documento combinando el título de la película (`movie_title`) como un ancla contextual rígida y el contenido de la crítica (`review_content`) como el texto rico en descriptores semánticos, generando la columna `text_combined`.

### Celda 5: Pipeline de Preprocesamiento de Texto
Aplica una limpieza de texto secuencial y profunda optimizada para modelos basados en contexto:
1.  Conversión estandarizada a minúsculas.
2.  Remoción de signos de puntuación, caracteres no ASCII y dígitos numéricos huérfanos.
3.  Eliminación de *stopwords* en inglés.
4.  **Lematización** (usando `WordNetLemmatizer`): Reduce las palabras a su raíz morfológica pero manteniéndolas legibles (evita la estemización o *stemming*) para que el backend de `SentenceTransformers` conserve la coherencia contextual completa del texto.

### Celda 6: Generación de Embeddings (Modelo Principal)
Extrae una muestra representativa y escalable de $10,000$ documentos para balancear la carga computacional y la memoria RAM. Carga el modelo **`all-MiniLM-L6-v2`** (384 dimensiones, ~80MB) por su sobresaliente velocidad en tareas de similitud semántica. Los vectores generados se normalizan bajo la norma L2 y se persisten en disco en un archivo binario indexado (`corpus_embeddings_minilm.npy`).

### Celda 7 & 8: Motor de Recuperación y Consultas Obligatorias
Define una función de búsqueda vectorizada (`search`) que codifica dinámicamente un query de entrada en lenguaje natural y computa la **similitud de coseno** con el corpus mediante el producto punto de matrices normalizadas. Se ejecutan y formatean con gradientes de color (*heatmaps* en tablas de Pandas) las 8 consultas obligatorias de evaluación ($Q1$ a $Q8$), extrayendo los $Top\text{-}5$ resultados por cada una.

### Celda 9: Tabla Resumen General
Itera sobre las matrices resultantes de las celdas previas y consolida de forma compacta el **Mejor Resultado Único ($Top\text{-}1$)** de cada consulta. Muestra el identificador del documento, el título de la película asociada y su puntuación de confianza semántica para contrastar rápidamente el rendimiento del modelo frente a diferentes géneros (Terror, Romance, Sci-Fi).

### Celda 11: Desafío de Excelencia — Visualización 2D con PCA
Evalúa visualmente la cohesión topológica del espacio de embeddings latente:
* Toma una muestra de 1,000 documentos y proyecta sus 384 dimensiones originales a un plano 2D usando **Análisis de Componentes Principales (PCA)**.
* Calcula y grafica la varianza explicada acumulada de los componentes PC1 y PC2.
* Pinta los puntos del espacio mediante un mapa de calor basado en la similitud hacia el query $Q1$ (*Science fiction*).
* **Corrección de Compatibilidad:** Utiliza un marcador nativo de estrella (`marker="*"`) para graficar de forma segura la ubicación espacial del query sin generar excepciones de codificación de fuentes en sistemas basados en Linux.

---

## 📈 Resumen Técnico del Modelo
* **Modelo de Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Dimensiones del Vector:** 384
* **Métricas de Cercanía:** Similitud Coseno (escalada de 0.0 a 1.0)
* **Algoritmo de Reducción:** PCA (Principal Component Analysis)
* **Visualizaciones Guardadas:** `pca_embeddings.png` (Gráfico de dispersión bi-panel: Similitud vs. Distribución General de Densidad Semántica).
