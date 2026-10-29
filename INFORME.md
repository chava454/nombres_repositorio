# Informe de evaluación: clasificación de imágenes CIFAR-10 mediante CNN

**Estudiante:** [Completar]  
**Matrícula:** [Completar en la copia de entrega, si corresponde]  
**Semilla:** 76  
**Asignatura:** Inteligencia Artificial  
**Institución:** Universidad Andina del Cusco

## Resumen

Se implementó una red neuronal convolucional (CNN) en PyTorch para clasificar imágenes RGB del dataset CIFAR-10 en diez categorías. El flujo incluye aumento de datos para entrenamiento, una partición estratificada de entrenamiento y validación, evaluación sobre el conjunto de prueba independiente y análisis mediante curvas de aprendizaje y matriz de confusión. La accuracy obtenida en prueba fue **73.37 %**.

## Objetivos

- Cargar CIFAR-10 y visualizar imágenes con sus etiquetas.
- Aplicar transformaciones y aumento de datos al conjunto de entrenamiento.
- Separar entrenamiento y validación de forma estratificada.
- Implementar y validar las dimensiones de entrada y salida de una CNN.
- Entrenar y evaluar el modelo e interpretar las métricas obtenidas.

## Fundamento teórico

Al aplanar una imagen CIFAR-10 se obtienen 3072 valores y se pierde la estructura espacial explícita de los píxeles. En un MLP, las capas densas conectan cada valor de entrada con numerosas neuronas, lo que aumenta el número de parámetros. Una CNN conserva la estructura espacial mediante filtros con conexiones locales y reutilización de pesos (*weight sharing*), reduciendo parámetros respecto de conexiones densas equivalentes. El pooling disminuye las dimensiones espaciales y ayuda a conservar características destacadas. Estas propiedades hacen que las CNN sean adecuadas para imágenes.

## Dataset y preparación

CIFAR-10 contiene 50 000 imágenes de entrenamiento y 10 000 de prueba, distribuidas en `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship` y `truck`. El conjunto de entrenamiento se dividió en 40 000 imágenes de entrenamiento y 10 000 de validación mediante partición estratificada. El conjunto de prueba se mantuvo independiente.

Para entrenamiento se usaron `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()` y normalización RGB. En validación y prueba se aplicaron conversión a tensor y normalización, sin aumento aleatorio. Se fijó la semilla 76 en PyTorch y NumPy.

## Arquitectura

La entrada es un tensor de forma `[B, 3, 32, 32]`, donde `B` es el tamaño del lote.

| Etapa | Operaciones | Forma de salida |
|---|---|---|
| Entrada | Imagen RGB | `[B, 3, 32, 32]` |
| Bloque 1 | Conv2d(3, 32, 3, padding=1), ReLU, MaxPool2d(2) | `[B, 32, 16, 16]` |
| Bloque 2 | Conv2d(32, 64, 3, padding=1), ReLU, MaxPool2d(2) | `[B, 64, 8, 8]` |
| Aplanado | Flatten | `[B, 4096]` |
| Capa densa | Linear(4096, 256), ReLU | `[B, 256]` |
| Salida | Linear(256, 10) | `[B, 10]` |

La prueba con un lote simulado de cuatro imágenes valida la salida `[4, 10]`.

## Configuración

| Parámetro | Valor |
|---|---:|
| Semilla | 76 |
| Tamaño de lote | 64 |
| Épocas | 8 |
| Optimizador | Adam |
| Tasa de aprendizaje | 0.001 |
| Función de pérdida | CrossEntropyLoss |
| Partición entrenamiento/validación | 80 % / 20 % |

## Resultados

| Época | Pérdida train | Accuracy train | Pérdida validación | Accuracy validación |
|---:|---:|---:|---:|---:|
| 1 | 1.4957 | 0.4576 | 1.2839 | 0.5414 |
| 2 | 1.1588 | 0.5851 | 1.1314 | 0.5944 |
| 3 | 1.0307 | 0.6338 | 1.0394 | 0.6321 |
| 4 | 0.9549 | 0.6634 | 0.9706 | 0.6619 |
| 5 | 0.8919 | 0.6872 | 0.9267 | 0.6710 |
| 6 | 0.8455 | 0.7059 | 0.8623 | 0.7011 |
| 7 | 0.8023 | 0.7204 | 0.8860 | 0.6896 |
| 8 | 0.7742 | 0.7264 | 0.8467 | 0.7030 |

**Accuracy final en prueba: 0.7337 (73.37 %).**

La pérdida de entrenamiento disminuyó en cada época y su accuracy aumentó. Las métricas de validación mejoraron en general, aunque en la época 7 la pérdida subió de 0.8623 a 0.8860 y la accuracy bajó de 0.7011 a 0.6896; ambas se recuperaron en la época 8. La diferencia final entre accuracy de entrenamiento y validación fue 0.0234 (2.34 puntos porcentuales). En conjunto, hay una oscilación puntual compatible con una señal moderada de sobreajuste, pero no una brecha grande al final. La accuracy de prueba de 73.37 % indica que el modelo clasificó correctamente alrededor de siete de cada diez imágenes de prueba.

## Artefactos

- [Muestras del dataset](outputs/dataset_samples.png)
- [Curvas de pérdida y accuracy](outputs/loss_accuracy.png)
- [Matriz de confusión](outputs/confusion_matrix.png)

## Ejecución

Desde la carpeta del proyecto:

```powershell
python src\model.py
python src\dataset.py
python src\train.py
```

La primera ejecución descarga CIFAR-10 y requiere conexión a internet.

## Repositorio

[https://github.com/chava454/nombres_repositorio](https://github.com/chava454/nombres_repositorio)
