# Tourist Trip Design Problem (TTDP) con Graph Embeddings y Graph Neural Networks

Este repositorio contiene el desarrollo de mi trabajo de grado de Maestría en Inteligencia Artificial, enfocado en resolver el **Tourist Trip Design Problem (TTDP)** con un enfoque innovador apoyado en Inteligencia Artificial, específicamente en **Graph Embeddings** y **Graph Neural Networks (GNNs)**.

## Descripción del Problema

El **Tourist Trip Design Problem (TTDP)** es un problema de optimización combinatoria que busca diseñar itinerarios turísticos óptimos considerando:

- **Puntos de Interés (POIs)**: Cada uno con coordenadas (x,y), tiempo de servicio, beneficio y ventanas de tiempo
- **Restricciones**: Ventanas de tiempo (open/close), tiempo de servicio, tiempo de viaje entre POIs
- **Objetivo**: Maximizar el beneficio total visitando la mayor cantidad de POIs factibles

## Enfoque Innovador: IA + Grafos

### **Visión General**
El proyecto busca llevar el TTDP más allá de las heurísticas clásicas, explorando si los **embeddings de grafos** y **modelos de grafos** pueden:

- **Representar mejor** las relaciones entre POIs
- **Mejorar la calidad** de las rutas (más beneficios en menos tiempo)
- **Facilitar la personalización** futura (adaptar a perfiles de usuarios)

### **Flujo de la Propuesta**

#### **1. Baseline Tradicional**
- **Heurística/Metaheurística**: Greedy / GRASP-VND que respete todas las restricciones
- **Propósito**: Punto de comparación para evaluar mejoras del enfoque de IA

#### **2. Graph Embeddings**
- **Técnica**: Node2Vec / DeepWalk sobre grafo construido con distancias y tiempos
- **Beneficio**: Captura relaciones más ricas que distancias puras:
  - Afinidad latente entre nodos
  - Roles estructurales
  - Comunidades naturales

#### **3. Integración de Embeddings en el Planificador**
- **Utilidad Híbrida**: Similitud latente + beneficio - costo (tiempo)
- **Optimización**: Combinar embeddings como señal de similitud en construcción de rutas
- **Eficiencia**: Pruning por k-NN latente para reducir complejidad

#### **4. Embeddings Semánticos/Heterogéneos**
- **Enriquecimiento**: Categorías de POI, zonas geográficas, preferencias
- **Resultado**: Embeddings más expresivos y contextuales

#### **5. Learning-to-Rank para Transiciones** 
- **Modelo**: MLP pequeño que aprende a puntuar transiciones u→v
- **Features**: Embeddings + distancia + compatibilidad de ventanas de tiempo
- **Objetivo**: Reforzar elección de rutas más prometedoras

#### **6. Extensión a Graph Neural Networks** 
- **Arquitectura**: GCN o GraphSAGE ligera para refinar embeddings
- **Inclusión**: Atributos de nodos y aristas
- **Potencial**: Salto hacia enfoque más profundo de IA

#### **7. Validación Experimental** 
- **Comparación**: Baseline vs. variantes con embeddings, semántica, link scorer, GNN
- **Métricas**: Score total, POIs visitados, factibilidad, tiempo de cómputo, estabilidad
- **Análisis**: Ablations para entender impacto de cada componente

#### **8. Consolidación** 
- **Sistema**: Pipeline reproducible (one-click)
- **Documentación**: Metodología, resultados y discusión
- **Futuro**: Limitaciones y líneas futuras (RL, datos reales, GNN temporales)

## Estructura del Repositorio

```
ttdp-trabajo-grado/
├── exact_pulp.py              # Solver exacto MILP (PuLP + CBC) - Referencia
├── baseline_greedy.py         # Algoritmo greedy de línea base
├── baseline_grasp_vnd.py      # Metaheurística GRASP-VND (baseline)
├── graph_embeddings/          # Implementación de embeddings de grafos
│   ├── node2vec_ttdp.py      # Node2Vec para POIs
│   ├── deepwalk_ttdp.py      # DeepWalk para POIs
│   └── graph_builder.py      # Construcción del grafo POI
├── neural_approaches/         # Enfoques de IA
│   ├── learning_to_rank.py   # MLP para scoring de transiciones
│   ├── gnn_ttdp.py          # Graph Neural Networks
│   └── hybrid_planner.py     # Planificador híbrido (embeddings + heurística)
├── data/                      # Instancias de prueba
│   ├── synthetic/            # 20 instancias hptoptw-j** (11,16,21 nodos)
│   └── real/                 # 6 benchmarks TOPTW de la literatura
├── experiments/               # Resultados experimentales
│   ├── configs/              # Configuraciones de experimentos
│   ├── logs/                 # Logs de ejecución
│   ├── tables/               # Tablas de resultados
│   └── figures/              # Gráficos y visualizaciones
├── notebooks/                 # Análisis y experimentos
│   ├── eda/                  # Análisis exploratorio de datos
│   ├── embeddings_analysis/  # Análisis de embeddings generados
│   ├── experiments/          # Experimentos comparativos
│   └── ablation_study/       # Estudio de ablación de componentes
└── references/               # Bibliografía y papers
```

## Instalación y Uso

### Requisitos
```bash
pip install pulp pandas numpy networkx node2vec torch torch-geometric scikit-learn matplotlib seaborn
```

### Ejecución del Baseline
```bash
# Algoritmo greedy
python baseline_greedy.py --file data/synthetic/hptoptw-j11a.csv --name hptoptw-j11a --runs 3

# Metaheurística GRASP-VND
python baseline_grasp_vnd.py --file data/synthetic/hptoptw-j11a.csv --name hptoptw-j11a
```

### Generación de Embeddings
```bash
# Node2Vec embeddings
python graph_embeddings/node2vec_ttdp.py --file data/synthetic/hptoptw-j11a.csv --dimensions 64

# DeepWalk embeddings
python graph_embeddings/deepwalk_ttdp.py --file data/synthetic/hptoptw-j11a.csv --dimensions 64
```

### Planificador Híbrido
```bash
# Planificador con embeddings
python neural_approaches/hybrid_planner.py --file data/synthetic/hptoptw-j11a.csv --embeddings node2vec --alpha 0.7
```

##  Formato de Datos

### Instancias Sintéticas (CSV)
```csv
id,x,y,s,p,a,b,f,instance,k,i,u,l
0,53,46,0,0,1,700,0.8,hptoptw-j11a,2,11,5,3
1,43,32,90,20,29,101,0.7,hptoptw-j11a,2,11,5,3
...
```

**Columnas**:
- `id`: Identificador del POI (0 = depósito)
- `x,y`: Coordenadas cartesianas
- `s`: Tiempo de servicio
- `p`: Beneficio/Profit
- `a`: Apertura de ventana de tiempo
- `b`: Cierre de ventana de tiempo
- `f`: Factor de importancia (opcional)

## Metodología de IA

### **1. Construcción del Grafo POI**
- **Nodos**: POIs con atributos (coordenadas, beneficios, ventanas de tiempo)
- **Aristas**: Conexiones basadas en distancias y compatibilidad temporal
- **Pesos**: Combinación de distancia euclidiana y restricciones de tiempo

### **2. Graph Embeddings**
- **Node2Vec**: Random walks con parámetros p,q para capturar roles estructurales
- **DeepWalk**: Random walks uniformes para representaciones latentes
- **Dimensión**: 64-128 vectores por POI

### **3. Learning-to-Rank**
- **Features de entrada**: 
  - Embeddings de POIs origen y destino
  - Distancia euclidiana
  - Compatibilidad de ventanas de tiempo
  - Tiempo de servicio
- **Salida**: Score de transición (0-1)
- **Entrenamiento**: Datos sintéticos generados por baseline

### **4. Graph Neural Networks**
- **Arquitectura**: GraphSAGE o GCN ligera
- **Capas**: 2-3 capas convolucionales
- **Pooling**: Global mean pooling para embeddings finales
- **Fine-tuning**: Ajuste de embeddings para tarea específica

### **5. Planificador Híbrido**
- **Combinación**: Utilidad tradicional + score de embeddings
- **Parámetro α**: Balance entre heurística clásica y similitud latente
- **Pruning**: k-NN en espacio de embeddings para reducir complejidad

##  Métricas de Evaluación

### **Calidad de Solución**
- **Score total**: Beneficio acumulado de la ruta
- **POIs visitados**: Número de puntos incluidos
- **Factibilidad**: Cumplimiento de todas las restricciones

### **Eficiencia Computacional**
- **Tiempo de cómputo**: Tiempo total de ejecución
- **Escalabilidad**: Comportamiento con instancias más grandes
- **Estabilidad**: Consistencia entre ejecuciones

### **Análisis de Embeddings**
- **Calidad de representación**: T-SNE visualizations
- **Similitud semántica**: Correlación con características de POIs
- **Transfer learning**: Aplicabilidad a nuevas instancias

## Próximos Pasos

### **Fase 1: Baseline y Embeddings** (En progreso)
1. ✅ Implementar baseline greedy y GRASP-VND
2. 🔄 Generar embeddings con Node2Vec/DeepWalk
3. 🔄 Análisis exploratorio de embeddings

### **Fase 2: Integración y Learning-to-Rank**
1. 🔄 Planificador híbrido con embeddings
2. 🔄 MLP para scoring de transiciones
3. 🔄 Validación experimental inicial

### **Fase 3: GNNs y Optimización**
1. 🔄 Implementación de GraphSAGE/GCN
2. 🔄 Fine-tuning de embeddings
3. 🔄 Optimización de hiperparámetros

### **Fase 4: Validación y Consolidación**
1. 🔄 Estudio de ablación completo
2. 🔄 Comparación con métodos del estado del arte
3. 🔄 Pipeline reproducible y documentación

## 📚 Referencias

### **TTDP y Optimización**
- **TTDP-Small**: [Instancias oficiales](https://jrmontoya.wordpress.com/research/instances/ttdp-small/)
- **TOPTW Benchmarks**: Conjunto de instancias reales de la literatura

### **Graph Embeddings y GNNs**
- **Node2Vec**: Grover, A., & Leskovec, J. (2016). Node2vec: Scalable feature learning for networks
- **DeepWalk**: Perozzi, B., et al. (2014). Deepwalk: Online learning of social representations
- **GraphSAGE**: Hamilton, W., et al. (2017). Inductive representation learning on large graphs
- **GCN**: Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks

### **Learning-to-Rank y Aplicaciones**
- **LTR en Grafos**: Aplicaciones de ranking en problemas de optimización combinatoria
- **Híbridos IA + Heurísticas**: Combinación de métodos clásicos y modernos

##  Contribución

Este es un proyecto académico de trabajo de grado. Para contribuciones o consultas, contactar al autor.

##  Licencia

Proyecto académico - Universidad Javeriana. Los algoritmos externos mantienen sus respectivas licencias originales.

---

**Desarrollado por**: Juan Sebastián Navas Gómez 
**Institución**: Pontificia Universidad Javeriana  
**Programa**: Maestría en Inteligencia Artificial  
**Año**: 2025  
**Enfoque**: Graph Embeddings + Graph Neural Networks para TTDP

### Notas de ejecución (Actualizado)
- Exacto multi‑ruta k (src):
  - CSV con k en el archivo: `python ttdp-trabajo-grado/src/exact_pulp_k_routes.py --file ttdp-trabajo-grado/data/synthetic/hptoptw-j11a.csv --name hptoptw-j11a --time-limit 60`
  - Forzar k por CLI: `python ttdp-trabajo-grado/src/exact_pulp_k_routes.py --file ttdp-trabajo-grado/data/synthetic/hptoptw-j11a.csv --name hptoptw-j11a --k 2 --time-limit 60`
- Requisitos para exacto k: `pulp`, además de `numpy` y `pandas` (y `openpyxl` si usas Excel en otros scripts).
