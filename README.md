# Federated Transfer Learning para MRI: FLAIR vs T1w

Este proyecto implementa un pipeline de **Federated Transfer Learning (FTL)** para clasificación binaria **Esclerosis Múltiple (EM) vs control** usando imágenes MRI. El objetivo es estudiar un escenario multicentro no-IID donde los hospitales no comparten exactamente la misma modalidad de entrada:

- Hospitales 1 y 2: `FLAIR`
- Hospitales 3 y 4: `T1w`

La etiqueta de entrenamiento se toma de `label_id` en los CSV (`1 = EM`, `0 = control`). Las máscaras del dataset no se usan para redefinir la tarea como lesión visible vs tejido sano.

## Estado Actual

El pipeline actual incluye:

- Simulación federada con **Flower** y **Ray**.
- Modelo **ResNet-18** adaptado a MRI en escala de grises.
- Estrategias comparadas:
  - `FedAvg` con agregación por media simple.
  - `FedProx` con término proximal real en el entrenamiento local.
- Split local por paciente para evitar fuga entre slices del mismo sujeto.
- Evaluación centralizada del modelo global sobre `global_test.csv`.
- Métricas a nivel de slice y a nivel de paciente.
- Baselines centralizado y local-only con resultados exportados a CSV.
- Logging automático de métricas por ronda en `results/raw/`.
- Generación de tablas y figuras para el informe LaTeX.

## Estructura del Proyecto

```text
proyecto_fl/
├── configs/
│   ├── experiment_fedavg.yaml       # Configuración FedAvg
│   ├── experiment_fedprox_mu_0p001.yaml
│   ├── experiment_fedprox_mu_0p01.yaml
│   └── experiment_fedprox_mu_0p1.yaml
│
├── data/
│   ├── MRIcontrol_kde/              # Imágenes control por modalidad
│   ├── MRIms_kde/                   # Imágenes EM por modalidad
│   ├── hospital_1_train.csv
│   ├── hospital_2_train.csv
│   ├── hospital_3_train.csv
│   ├── hospital_4_train.csv
│   ├── global_test.csv              # Test centralizado sin solape de pacientes
│   ├── hospital_summary.csv
│   ├── patient_assignment.csv
│   ├── project_focus.txt
│   └── README.md
│
├── results/
│   ├── raw/                         # Métricas por ronda de cada experimento
│   ├── tables/                      # Tablas CSV para el informe
│   └── figures/                     # Figuras PNG para el informe
│
├── src/
│   ├── baseline.py                  # Baselines centralizado/local
│   ├── client_app.py                # Flower ClientApp
│   ├── client.py                    # Cliente NumPyClient y entrenamiento local
│   ├── dataset.py                   # Dataset, DataLoaders y split por paciente
│   ├── metrics.py                   # Accuracy, F1, sensibilidad, especificidad
│   ├── model.py                     # ResNet-18 adaptada a 1 canal
│   ├── plot_results.py              # Generación de tablas y figuras
│   ├── server_app.py                # Flower ServerApp y evaluación global
│   └── server.py                    # Estrategia de agregación simple mean
│
├── pyproject.toml                   # Dependencias y configuración Flower App
├── Federated_Learning_Project_Report.pdf
├── README.md
└── run_sim.py                       # Lanzador principal de simulación
```

## Requisitos

Entorno recomendado:

- Python 3.10+
- Conda
- PyTorch con CUDA
- Flower
- Ray
- WSL2 en Windows
- GPU NVIDIA compatible con CUDA

El proyecto se ha probado con una **NVIDIA RTX 4060** simulando 4 clientes con `num_gpus: 0.25` por cliente.

## Informe

La versión preparada para entrega está en:

```text
Federated_Learning_Project_Report.pdf
```

El informe usa las figuras de `results/figures/` y las tablas derivadas de `results/tables/`. Los resultados numéricos del README proceden de los CSV generados tras volver a ejecutar los experimentos completos el 17/05/2026.

## Instalación

Desde WSL:

```bash
conda activate fl_mri
cd /home/alpha/proyecto_fl
pip install -e .
```

Si el entorno no existe:

```bash
conda create -n fl_mri python=3.10
conda activate fl_mri
pip install -e .
```

## Ejecución de Experimentos

El lanzador principal es `run_sim.py`. Se usa la API de simulación de Flower porque permite controlar de forma directa los recursos Ray/GPU en WSL2.

### FedAvg

```bash
python run_sim.py
```

Por defecto, `run_sim.py` usa:

```text
configs/experiment_fedavg.yaml
```

### FedProx

```bash
FL_CONFIG_PATH=configs/experiment_fedprox_mu_0p01.yaml python run_sim.py
```

También se puede pasar la configuración por argumento:

```bash
python run_sim.py --config configs/experiment_fedprox_mu_0p01.yaml
```

Para el barrido rápido de FedProx:

```bash
python run_sim.py --config configs/experiment_fedprox_mu_0p001.yaml
python run_sim.py --config configs/experiment_fedprox_mu_0p01.yaml
python run_sim.py --config configs/experiment_fedprox_mu_0p1.yaml
```

### Monitorización de GPU

```bash
watch -n 1 nvidia-smi
```

## Baselines

Para ejecutar todos los baselines mínimos y guardar sus métricas:

```bash
python -m src.baseline --config configs/experiment_fedavg.yaml --all-baselines
```

Esto entrena:

- `centralized_all`: todos los hospitales juntos.
- `local_only_FLAIR`: solo hospitales FLAIR.
- `local_only_T1w`: solo hospitales T1w.
- `local_only_hospital_X`: cada hospital aislado.

Los resultados se guardan como:

```text
results/raw/baselines_<run_id>_metrics.csv
```

## Configuración Experimental Final

Los experimentos principales del informe usan:

```yaml
federated_learning:
  num_rounds: 10

client_training:
  local_epochs: 2
  learning_rate: 0.0001
  optimizer: "Adam"
```

Para FedProx:

```yaml
federated_learning:
  strategy: "FedProx"
  proximal_mu: 0.01
```

## Evaluación

Hay dos evaluaciones:

- **Centralizada en servidor**: usa `global_test.csv`. Es la evaluación principal del informe.
- **Distribuida/local**: Flower evalúa cada cliente sobre su validación local.

El test global no comparte pacientes con los CSV de entrenamiento. Las métricas centralizadas se guardan automáticamente en:

```text
results/raw/
```

Las métricas principales se reportan a dos niveles:

- **Slice**: cada corte axial cuenta como una muestra.
- **Paciente**: se promedian las probabilidades de clase EM de todos los slices de un paciente y se aplica umbral 0.5.

Ejemplos de archivos de salida:

```text
results/raw/
├── FTL_FLAIR_vs_T1w_FedAvg_FedAvg_<run_id>_metrics.csv
├── FTL_FLAIR_vs_T1w_FedProx_mu_0p001_FedProx_mu_0p001_<run_id>_metrics.csv
├── FTL_FLAIR_vs_T1w_FedProx_mu_0p01_FedProx_mu_0p01_<run_id>_metrics.csv
├── FTL_FLAIR_vs_T1w_FedProx_mu_0p1_FedProx_mu_0p1_<run_id>_metrics.csv
└── baselines_<run_id>_metrics.csv
```

## Resultados Finales

Resultados sobre `global_test.csv` tras 10 rondas. Se reportan métricas a nivel de slice y a nivel de paciente:

| Estrategia | Ronda | Loss | Acc. slice | F1 slice | Sens. slice | Esp. slice | Acc. paciente | F1 paciente |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 10 | 0.0426 | 0.9892 | 0.9892 | 0.9780 | 0.9987 | 1.0000 | 1.0000 |
| FedProx (`mu=0.001`) | 10 | 0.0303 | 0.9892 | 0.9892 | 0.9975 | 0.9822 | 1.0000 | 1.0000 |
| FedProx (`mu=0.01`) | 10 | 0.0223 | **0.9924** | **0.9924** | 0.9885 | 0.9958 | 1.0000 | 1.0000 |
| FedProx (`mu=0.1`) | 10 | 0.0749 | 0.9755 | 0.9755 | 0.9560 | 0.9920 | 0.9792 | 0.9791 |

Mejor ronda observada por F1 global:

| Estrategia | Mejor ronda | Acc. slice | F1 slice | Sens. slice | Esp. slice | F1 paciente |
|---|---:|---:|---:|---:|---:|---:|
| FedAvg | 9 | 0.9929 | 0.9929 | 0.9875 | 0.9975 | 1.0000 |
| FedProx (`mu=0.001`) | 9 | 0.9899 | 0.9899 | 0.9870 | 0.9924 | 1.0000 |
| FedProx (`mu=0.01`) | 10 | **0.9924** | **0.9924** | 0.9885 | 0.9958 | 1.0000 |
| FedProx (`mu=0.1`) | 7 | 0.9773 | 0.9773 | 0.9600 | 0.9920 | 0.9791 |

Resumen de baselines principales sobre `global_test.csv`:

| Baseline | Acc. slice | F1 slice | Acc. paciente | F1 paciente |
|---|---:|---:|---:|---:|
| Centralizado (`centralized_all`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Local-only FLAIR | 0.7848 | 0.7786 | 0.7500 | 0.7393 |
| Local-only T1w | 0.7500 | 0.7266 | 0.7500 | 0.7266 |

Lectura principal:

- `FedProx` con `mu=0.01` obtiene el mejor rendimiento final a nivel de slice entre los experimentos federados.
- `FedAvg`, `FedProx mu=0.001` y `FedProx mu=0.01` alcanzan F1 de paciente igual a 1.0000 en el test global.
- `FedProx mu=0.1` resulta demasiado restrictivo en este reparto y reduce especialmente la sensibilidad.
- Los baselines local-only quedan claramente por debajo de los federados, lo que apoya la utilidad de colaborar entre hospitales y modalidades.
- El baseline centralizado actúa como cota superior experimental, pero pierde la restricción de privacidad/descentralización que motiva el aprendizaje federado.

## Generación de Tablas y Figuras

Después de ejecutar los experimentos:

```bash
python -m src.plot_results
```

El script genera:

```text
results/
├── figures/
│   ├── methodology_diagram.png
│   ├── dataset_partition.png
│   ├── class_distribution_by_modality.png
│   ├── fedavg_vs_fedprox_accuracy.png
│   ├── fedavg_vs_fedprox_f1.png
│   ├── fedavg_vs_fedprox_patient_f1.png
│   └── modality_comparison.png
│
└── tables/
    ├── dataset_distribution.csv
    ├── patient_distribution.csv
    ├── experiment_config.csv
    ├── experiment_summary.csv
    ├── baseline_summary.csv
    ├── method_comparison.csv
    ├── per_modality_metrics.csv
    ├── per_modality_patient_metrics.csv
    ├── per_hospital_metrics.csv
    ├── per_hospital_patient_metrics.csv
    └── baseline_patient_modality_metrics.csv
```

Estas figuras están pensadas para insertarse directamente en el informe LaTeX.

## Detalles Técnicos

### Modelo

`src/model.py` define una ResNet-18 de `torchvision`:

- Primera convolución adaptada de 3 canales a 1 canal.
- Capa final adaptada a 2 clases.
- Inicialización de la primera capa mediante promedio de pesos RGB preentrenados.

### Datos

`src/dataset.py`:

- Lee los CSV de cada hospital.
- Usa `label_id` como etiqueta EM/control.
- Hace split local train/val por `patient_id`.
- Carga `global_test.csv` para evaluación centralizada.

### Cliente

`src/client.py`:

- Entrena localmente con Adam y CrossEntropyLoss.
- Implementa el término proximal de FedProx:

```text
loss = CE + (mu / 2) * ||w_local - w_global||^2
```

### Servidor

`src/server.py`:

- Implementa agregación por media simple.
- No pondera por número de muestras.

`src/server_app.py`:

- Carga configuración YAML.
- Configura FedAvg/FedProx.
- Evalúa el modelo global en `global_test.csv`.
- Guarda métricas por ronda.

## Notas sobre Flower CLI

El `pyproject.toml` está preparado como Flower App, pero para este proyecto se usa `run_sim.py` porque ofrece control directo sobre:

- `RAY_NODE_IP_ADDRESS=127.0.0.1`
- `num_supernodes=4`
- `client_resources={"num_cpus": 2, "num_gpus": 0.25}`

Flower avisa de que `run_simulation` está deprecado, pero para esta simulación local en WSL2 resulta práctico y estable.

## Autores

- Álvaro Pastor García
- Dmytro Morgun
- Jorge Serrano Jiménez

Universidad de Málaga, ETSI Informática.
