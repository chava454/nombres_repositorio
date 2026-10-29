# Informe de evaluación: clasificación de imágenes CIFAR-10 mediante CNN

**Estudiante:** [Escribe tu nombre]  
**Código de matrícula:** [Escribe tu código]  
**Semilla utilizada:** 76 *(dos últimos dígitos de la matrícula)*  
**Asignatura:** Inteligencia Artificial  
**Institución:** Universidad Andina del Cusco  
**Fecha:** [Escribe la fecha de entrega]

> **Pendiente antes de entregar:** completar la sección de resultados con la salida real de `python src\train.py`, revisar que las figuras existan y reemplazar los campos entre corchetes. No se deben presentar valores estimados como resultados experimentales.

## Resumen

Este trabajo implementa un clasificador de imágenes para CIFAR-10 mediante una red neuronal convolucional (CNN) construida con PyTorch. El flujo incluye la carga y exploración visual del dataset, transformaciones de imágenes, una partición estratificada de entrenamiento y validación, la definición de la arquitectura y la evaluación sobre el conjunto de prueba independiente. El entrenamiento utiliza la función de pérdida CrossEntropyLoss y el optimizador Adam.

La precisión final obtenida en el conjunto de prueba fue **[completar] %**. El análisis de las curvas de aprendizaje y de la matriz de confusión se completa con los resultados generados durante la ejecución.

## 1. Introducción

La clasificación de imágenes consiste en asignar una etiqueta a una imagen de acuerdo con su contenido visual. CIFAR-10 es un dataset de referencia compuesto por imágenes RGB de 32 × 32 píxeles distribuidas en diez categorías. Debido a que las imágenes contienen variaciones de posición, forma, color y fondo, el problema permite observar cómo una CNN aprende características visuales a partir de ejemplos etiquetados.

El objetivo de esta evaluación es preparar los datos de manera reproducible, implementar una CNN con dimensiones compatibles y analizar su desempeño mediante métricas de entrenamiento, validación y prueba.

## 2. Objetivos

### Objetivo general

Desarrollar y evaluar una CNN capaz de clasificar imágenes del dataset CIFAR-10.

### Objetivos específicos

- Cargar CIFAR-10 y visualizar ejemplos con sus etiquetas.
- Aplicar normalización y aumento de datos al conjunto de entrenamiento.
- Separar el conjunto de entrenamiento original en subconjuntos de entrenamiento y validación mediante muestreo estratificado.
- Implementar y comprobar el flujo de tensores 4D de una CNN.
- Entrenar el modelo y evaluar su desempeño sobre el conjunto de prueba.
- Interpretar las curvas de aprendizaje y la matriz de confusión a partir de los resultados observados.

## 3. Fundamento teórico

Una imagen RGB de CIFAR-10 tiene 32 × 32 × 3 = 3072 valores por imagen. Un MLP puede recibir esos valores aplanados, pero pierde explícitamente la disposición espacial de los píxeles. Además, las capas densas conectan cada entrada con muchas neuronas, lo que puede incrementar el número de parámetros y el costo computacional.

Las CNN aprovechan la estructura espacial de las imágenes mediante:

- **Conexiones locales:** los filtros analizan vecindarios pequeños y detectan patrones como bordes o texturas.
- **Compartición de pesos (weight sharing):** cada filtro reutiliza sus parámetros en distintas ubicaciones, reduciendo el número de parámetros frente a conexiones densas equivalentes.
- **Submuestreo:** MaxPool2d reduce las dimensiones espaciales y conserva respuestas destacadas.
- **Tolerancia a desplazamientos pequeños:** las convoluciones y el pooling ayudan a reconocer patrones aunque cambien moderadamente de posición. Esto no implica invariancia perfecta.

Por estas razones, una CNN conserva y aprovecha mejor la estructura de una imagen que un MLP que la aplana desde el inicio.

## 4. Dataset y preparación de los datos

CIFAR-10 incluye las clases `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship` y `truck`. El conjunto oficial de entrenamiento contiene 50 000 imágenes y el conjunto oficial de prueba contiene 10 000.

El proyecto separa el conjunto oficial de entrenamiento en entrenamiento y validación, conservando la proporción de las clases mediante `train_test_split` con `stratify`. Con una partición de validación del 20 %, se esperan 40 000 imágenes para entrenamiento y 10 000 para validación; el conjunto de prueba permanece independiente con 10 000 imágenes.

Las transformaciones para entrenamiento incluyen:

1. `RandomCrop(32, padding=4)` para generar recortes aleatorios con relleno.
2. `RandomHorizontalFlip()` para invertir horizontalmente algunas imágenes.
3. `ToTensor()` para convertir las imágenes a tensores.
4. `Normalize(...)` para normalizar los tres canales RGB.

Para validación y prueba se usa conversión a tensor y normalización, sin aumento aleatorio. La cuadrícula de ejemplos se genera en `outputs/dataset_samples.png`.

## 5. Arquitectura de la CNN

La entrada del modelo tiene la forma **[B, 3, 32, 32]**, donde `B` es el tamaño del lote, `3` representa los canales RGB y las dimensiones espaciales son 32 × 32.

| Etapa | Operación | Forma de salida para una entrada `[B, 3, 32, 32]` |
|---|---|---|
| Entrada | Imagen RGB | `[B, 3, 32, 32]` |
| Bloque 1 | Conv2d(3, 32, 3, padding=1) + ReLU + MaxPool2d(2) | `[B, 32, 16, 16]` |
| Bloque 2 | Conv2d(32, 64, 3, padding=1) + ReLU + MaxPool2d(2) | `[B, 64, 8, 8]` |
| Aplanado | Flatten | `[B, 4096]` |
| Capa densa | Linear(4096, 256) + ReLU | `[B, 256]` |
| Salida | Linear(256, 10) | `[B, 10]` |

La salida contiene diez logits por imagen, uno para cada clase. CrossEntropyLoss recibe estos logits junto con las etiquetas enteras; no se aplica Softmax explícitamente antes de la pérdida.

La prueba de formas de `src/model.py` debe mostrar, para un lote simulado de cuatro imágenes, una entrada `[4, 3, 32, 32]`, una salida convolucional `[4, 64, 8, 8]` y una salida final `[4, 10]`.

## 6. Configuración experimental

| Parámetro | Valor |
|---|---:|
| Semilla | 76 |
| Tamaño del lote | 64 |
| Número de épocas | 8 |
| Optimizador | Adam |
| Tasa de aprendizaje | 0.001 |
| Función de pérdida | CrossEntropyLoss |
| Dispositivo | [CPU o CUDA, según la salida del programa] |
| Partición entrenamiento/validación | 80 % / 20 % del conjunto oficial de entrenamiento |

La semilla 76 corresponde a los dos últimos dígitos de la matrícula y se aplica a PyTorch y NumPy para favorecer la reproducibilidad de la inicialización, la partición y el orden aleatorio de los datos.

## 7. Resultados

Completar esta sección después de que `python src\train.py` termine correctamente. Copiar las métricas de la terminal y revisar las figuras guardadas en `outputs/`.

### Métricas por época

| Época | Pérdida train | Accuracy train | Pérdida validación | Accuracy validación |
|---:|---:|---:|---:|---:|
| 1 | [ ] | [ ] | [ ] | [ ] |
| 2 | [ ] | [ ] | [ ] | [ ] |
| 3 | [ ] | [ ] | [ ] | [ ] |
| 4 | [ ] | [ ] | [ ] | [ ] |
| 5 | [ ] | [ ] | [ ] | [ ] |
| 6 | [ ] | [ ] | [ ] | [ ] |
| 7 | [ ] | [ ] | [ ] | [ ] |
| 8 | [ ] | [ ] | [ ] | [ ] |

**Accuracy final en prueba:** [completar con el valor `Test Accuracy` de la terminal]  
**Mejor accuracy de validación:** [completar]  
**Época con mejor validación:** [completar]

### Artefactos gráficos

- `outputs/dataset_samples.png`: muestra ejemplos del dataset y sus etiquetas.
- `outputs/loss_accuracy.png`: compara pérdida y accuracy de entrenamiento y validación por época.
- `outputs/confusion_matrix.png`: muestra las predicciones del conjunto de prueba frente a las etiquetas reales.

## 8. Análisis de resultados

Completar este análisis después de observar las métricas y las gráficas:

- **Aprendizaje:** describir si la pérdida de entrenamiento disminuye y si la accuracy aumenta durante las épocas.
- **Generalización:** comparar entrenamiento y validación. Una diferencia grande y creciente, en especial si empeora la pérdida de validación mientras mejora la de entrenamiento, puede indicar sobreajuste.
- **Subajuste:** si las métricas de entrenamiento y validación permanecen ambas bajas y las pérdidas altas, el modelo podría no haber aprendido suficientemente los patrones.
- **Prueba:** interpretar `Test Accuracy` como la proporción de imágenes correctamente clasificadas en el conjunto de prueba independiente.
- **Matriz de confusión:** identificar las clases con más aciertos y los pares de categorías que más se confunden. Las cifras de la diagonal principal son aciertos.

**Análisis con valores observados:**  
[Escribe aquí la comparación cuantitativa entre la última época de entrenamiento y validación, el resultado de prueba y las confusiones visibles en la matriz. No concluyas sobre overfitting o underfitting hasta revisar esos datos.]

## 9. Conclusiones

La evaluación permitió implementar el flujo de preparación de CIFAR-10, una CNN de dos bloques convolucionales y la validación explícita de las formas de los tensores. El resultado final del clasificador fue **[completar] %** de accuracy sobre prueba. De acuerdo con las curvas y la matriz de confusión, **[completar conclusión basada en los resultados reales]**.

## 10. Cómo ejecutar el proyecto

Desde la carpeta `nombres_repositorio`, ejecutar en PowerShell:

```powershell
python src\model.py
python src\dataset.py
python src\train.py
```

La primera ejecución descarga CIFAR-10 y requiere conexión a internet. El entrenamiento puede tardar varios minutos, especialmente si se ejecuta en CPU.

## 11. Repositorio

**Enlace público de GitHub:** https://github.com/chava454/nombres_repositorio

Antes de entregar, verificar que el repositorio sea público y que incluya el código, el notebook, el informe, el README y los artefactos gráficos requeridos. No subir el dataset descargado salvo que el docente lo solicite.
