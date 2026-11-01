# ACI606 - Cloud Fleet Challenge

Servicio Cloud persistente y en vivo para **ingesta de telemetría**, **inferencia en tiempo real con Machine Learning**, **observabilidad** y **panel de control de flota de robots**.

---

## Tabla de contenidos

- [URL pública del servicio](#-url-pública-del-servicio-desplegado-render)
- [Arquitectura del sistema](#️-arquitectura-del-sistema)
- [Evidencia del modelo de Machine Learning](#-evidencia-del-modelo-de-machine-learning)
- [Variables de entorno](#️-variables-de-entorno-requeridas)
- [Guía de ejecución local](#-guía-de-ejecución-local)
- [Pruebas de humo](#-pruebas-de-humo-smoke-tests)

---

## 🔗 URL pública del servicio desplegado (Render)

| Recurso | URL |
|---|---|
| **URL Base** | https://aci606-cloud-fleet-challenge.onrender.com |
| **Dashboard** | https://aci606-cloud-fleet-challenge.onrender.com/dashboard |
| **OpenAPI / Swagger** | https://aci606-cloud-fleet-challenge.onrender.com/docs |
| **Health Check** | https://aci606-cloud-fleet-challenge.onrender.com/health |
| **Metrics** | https://aci606-cloud-fleet-challenge.onrender.com/metrics |

---

## 🏗️ Arquitectura del sistema

El sistema implementa una arquitectura **PaaS desacoplada, persistente y orientada a la observabilidad**:

- **Servicio Web (PaaS):** desplegado en **Render**, ejecutando la API construida con **FastAPI** y la interfaz ASGI **Uvicorn**.
- **Almacenamiento persistente:** base de datos **PostgreSQL** gestionada en **Neon.tech**, conectada mediante **SQLAlchemy** (driver `psycopg3` y `pool_pre_ping=True` para prevenir desconexiones tras periodos de inactividad).
- **Servicio de inferencia ML:** pipeline cargado al inicio del servicio (`models/model.joblib`), que evalúa en tiempo real las muestras enviadas al endpoint `POST /predict`.
- **Observabilidad y dashboard:** cobertura de métricas operacionales (solicitudes, errores, latencia promedio y tiempo de actividad) y vista web en `/dashboard`.

---

## 📊 Evidencia del modelo de Machine Learning

Se implementó una comparación entre dos clasificadores basados en **Regresión Logística con Regularización** (Ridge vs Lasso) utilizando `scikit-learn`.

### 1. Análisis Exploratorio de Datos (EDA)

Distribución de clases en el dataset generado ($N = 3{,}000$):

| Clase | Muestras |
|---|---:|
| `CRITICAL` | 2,153 |
| `WARNING` | 534 |
| `NORMAL` | 313 |

### 2. Estrategia de evaluación

- **División de datos:** `GroupShuffleSplit` basado en `robot_id` (75% entrenamiento / 25% prueba), para garantizar la evaluación sobre robots no vistos durante el entrenamiento.
- **Métrica principal:** **F1 Macro**, que otorga el mismo peso a todas las clases independientemente de su frecuencia:

$$
F1_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} F1_k
$$

### 3. Comparación de modelos

| Modelo | Penalización | Solver | F1 Macro | Estado |
|---|:---:|:---:|:---:|---|
| Logistic Regression (Ridge) | $L_2$ | `lbfgs` | 0.6423 | ✅ **Seleccionado** |
| Logistic Regression (Lasso) | $L_1$ | `saga` | 0.6423 | Evaluado |

### 4. Matriz de confusión

Modelo seleccionado: **Ridge (L2)**.

| Real \ Predicción | Pred NORMAL | Pred WARNING | Pred CRITICAL |
|---|:---:|:---:|:---:|
| **Real NORMAL** | 0 | 0 | 0 |
| **Real WARNING** | 1 | 33 | 4 |
| **Real CRITICAL** | 0 | 0 | 712 |

---

## 🛠️ Variables de entorno requeridas

En la plataforma Render se configuraron las siguientes variables de entorno:

| Variable | Descripción | Ejemplo / Valor |
|---|---|---|
| `DATABASE_URL` | Cadena de conexión PostgreSQL de Neon | `postgresql+psycopg://user:pass@ep-xyz.neon.tech/neondb?sslmode=require` |
| `MODEL_PATH` | Ruta al pipeline binario del modelo | `models/model.joblib` |

---

## 🚀 Guía de ejecución local

### Paso A: Crear y activar entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Paso B: Generar dataset y entrenar el modelo

```bash
python simulator/generate_dataset.py --samples 3000
python ml/train_model.py
```

### Paso C: Configurar variable de entorno e iniciar la API

```bash
export DATABASE_URL="postgresql://usuario:password@ep-xyz.neon.tech/neondb?sslmode=require"
uvicorn app.main:app --reload
```

Abre en tu navegador:

- Dashboard: http://127.0.0.1:8000/dashboard
- Swagger: http://127.0.0.1:8000/docs

### Paso D: Conectar el simulador de la flota virtual

En otra pestaña de la terminal (con el entorno `.venv` activo):

```bash
python simulator/fleet_simulator.py --api http://127.0.0.1:8000 --robots 3
```

---

## 🧪 Pruebas de humo (Smoke Tests)

Para verificar el cumplimiento del contrato del servicio, local o remoto:

```bash
# Prueba en entorno local
python tests/smoke_test.py --api http://127.0.0.1:8000

# Prueba en entorno remoto (Render)
python tests/smoke_test.py --api https://aci606-cloud-fleet-challenge.onrender.com
```
