# Evaluación Parcial N°2 — Deep Learning (DUOC UC)

Clasificación de personajes de **Los Simpson** mediante *Convolutional Neural Networks* (CNN) con **Keras / TensorFlow**.

- **Asignatura:** Deep Learning — 2° semestre 2026
- **Integrantes:** Esteban Gamboa · Felipe Ortega
- **Entrega:** vía GitHub

---

## 1. Objetivo

Entrenar un clasificador de imágenes basado en CNN capaz de reconocer **18 personajes** de Los Simpson, alcanzando una **accuracy en el test set ≥ 85 %**.

---

## 2. Dataset

Imágenes de personajes extraídas de capítulos de la serie, recopiladas por [Alexandre Attia](http://www.alexattia.fr/). Es un dataset más complejo que Fashion MNIST: los personajes aparecen en distintas poses, posiciones y con otros personajes en pantalla.

| Archivo | Tamaño | Contenido |
|---|---|---|
| `simpsons_train.tar.gz` | ~500 MB | Imágenes de entrenamiento (una subcarpeta por personaje) |
| `simpsons_test.tar.gz` | ~10 MB | Imágenes de test (nombre con prefijo del personaje) |

> ⚠️ **Git LFS:** los `.tar.gz` se almacenan con Git LFS. Tras clonar, ejecuta `git lfs pull` para descargar el contenido real (de lo contrario solo tendrás los punteros).

**Las 18 clases utilizadas** (las de mayor número de imágenes):

| # | Personaje | # | Personaje |
|---|---|---|---|
| 0 | abraham_grampa_simpson | 9 | krusty_the_clown |
| 1 | apu_nahasapeemapetilon | 10 | lisa_simpson |
| 2 | bart_simpson | 11 | marge_simpson |
| 3 | charles_montgomery_burns | 12 | milhouse_van_houten |
| 4 | chief_wiggum | 13 | moe_szyslak |
| 5 | comic_book_guy | 14 | ned_flanders |
| 6 | edna_krabappel | 15 | nelson_muntz |
| 7 | homer_simpson | 16 | principal_skinner |
| 8 | kent_brockman | 17 | sideshow_bob |

---

## 3. Estructura del repositorio

```
DeepLearningEV2/
├── README.md                    ← este archivo
├── CNN_SIMPSON_(1).ipynb        ← notebook con todo el análisis y entrenamiento
├── .gitattributes               ← configura Git LFS (*.tar.gz)
├── simpsons_train.tar.gz        ← dataset de entrenamiento (LFS)
└── simpsons_test.tar.gz         ← dataset de test (LFS)
```

---

## 4. Tecnologías

- **Python 3**, **TensorFlow 2.21.0** y **Keras 3.13.2**
- **OpenCV** (`cv2`) — lectura y redimensionado de imágenes
- **NumPy**, **Pandas**, **Matplotlib**, **scikit-learn**
- **Google Colab** con **GPU** (recomendado por el costo computacional)
- **Git LFS** para versionar los archivos grandes del dataset

---

## 5. Metodología

1. **Análisis de datos:** revisión de la distribución de clases (dataset desbalanceado) y de muestras de imágenes.
2. **Preprocesamiento:**
   - Redimensionado de todas las imágenes a **64 × 64**.
   - Normalización de píxeles a **[0, 1]** (originalmente 0–255).
   - Partición **80 / 20 train / validation estratificada**.
   - Cálculo de **`class_weight` balanceados** para compensar el desbalance de clases.
3. **Entrenamiento y comparación de modelos** (ver tabla).
4. **Evaluación final** del mejor modelo sobre el test set: accuracy global, *precision*, *recall* y *F1* por clase, matriz de confusión y análisis visual de errores.

**Hiperparámetros base:** `batch_size = 64`, optimizador **Adam** (`lr = 1e-3`), pérdida `sparse_categorical_crossentropy`, semilla fija `SEED = 42` para reproducibilidad.

---

## 6. Modelos comparados

| # | Modelo | Características |
|---|---|---|
| 1 | **MLP (Fully Connected)** | Línea base: aplana la imagen y usa capas densas (512 → 256) con `Dropout`. |
| 2 | **CNN simple** | Dos bloques `Conv2D` + `MaxPooling` y una capa densa. Sin regularización fuerte. |
| 3 | **CNN profunda** | Estilo VGG: bloques de 2 convoluciones 3×3 + `BatchNormalization` + `MaxPooling` + `Dropout`, cabeza densa regularizada. |
| 4 | **CNN profunda + Data Augmentation** ⭐ | Igual que (3) más aumento de datos en línea: `RandomFlip` (horizontal), `RandomRotation`, `RandomZoom` y `RandomTranslation`. **Modelo final.** |

El modelo final usa los *callbacks* `ReduceLROnPlateau`, `EarlyStopping` y `ModelCheckpoint`.

---

## 7. Cómo ejecutar

> El notebook está pensado para **Google Colab** con GPU.

1. Clonar el repositorio y descargar el dataset:
   ```bash
   git clone https://github.com/ealejandro23/DeepLearningEV2.git
   cd DeepLearningEV2
   git lfs pull
   ```
2. Subir `CNN_SIMPSON_(1).ipynb` y los `.tar.gz` a tu Google Drive (por ejemplo a `MyDrive/eva 2 deep learning`).
3. Abrir el notebook en Colab y activar **Entorno de ejecución → Cambiar tipo de entorno → GPU**.
4. Ejecutar las celdas en orden. Las rutas de datos se configuran en la celda de carga (actualmente apuntan a `/content/drive/MyDrive/eva 2 deep learning`).

**Requisitos:** `tensorflow`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

---

## 8. Resultados

Las métricas se calculan en las secciones **7 a 9** del notebook: comparación de modelos, reporte por clase y matriz de confusión.

**Hallazgos principales:**

- La **CNN supera claramente al MLP**: la red totalmente conectada no aprovecha la estructura espacial de la imagen y generaliza peor.
- La **profundidad + regularización** (`BatchNormalization` y `Dropout`) reduce la brecha entre train y validación y estabiliza el entrenamiento.
- El **data augmentation** aporta variedad de poses y encuadres, reduciendo el *overfitting*.
- Los **`class_weight`** evitan que el modelo favorezca a los personajes con más imágenes (p. ej., Homer).
- La **matriz de confusión** y las imágenes mal clasificadas permiten identificar qué personajes se confunden entre sí.

> 💡 El test set contiene imágenes algo más "fáciles" que las de entrenamiento, por lo que sus métricas suelen ser mejores.

---

## 9. Posibles mejoras

- Usar imágenes de mayor resolución (96 × 96 o 128 × 128).
- Aplicar **transfer learning** (MobileNetV2, ResNet50).
- Modelos tipo **ensemble**.

---

## 10. Licencia y uso académico

Proyecto realizado con fines **académicos** para la asignatura de Deep Learning (DUOC UC). El dataset pertenece a su autor original, Alexandre Attia.
