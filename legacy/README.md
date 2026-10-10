# Archivo legacy

Todo el contenido de este directorio es histórico y no productivo. No forma parte de la arquitectura operativa actual ni constituye una segunda aplicación mantenida.

SIAR Sync actual está en [`/siar-sync`](../siar-sync/README.md). Los scripts archivados aquí no deben ejecutarse contra los datos canónicos sin una revisión previa.

Este material conserva código, interfaces, dependencias y artefactos de etapas anteriores. Puede contener rutas absolutas de Windows, nombres de archivos antiguos, contratos de API que ya no estén vigentes, rutas relativas que dependían de la ubicación original y referencias documentales históricas. Algunas referencias de `docs_1.*` y `docs_2.*` conservan nombres o rutas de entonces deliberadamente.

Los resultados derivados asociados a estos scripts se clasificarán y moverán en una fase independiente a `legacy/data/siar-derived/`. Esta reorganización de código no mueve datos ni cambia los resultados existentes.

La presencia de un script o resultado en `legacy/` no significa que su método haya sido validado como baseline científico actual. En una fase posterior, cualquier lógica que se considere útil deberá extraerse a código nuevo, reproducible y probado.

## Contenido

- `acquisition/`: clientes y utilidades históricas de adquisición y catálogo.
- `et0/`: scripts históricos de depuración, cálculo ET₀ y análisis de errores.
- `ml/`: experimentos y entrenamiento histórico de redes neuronales.
- `dashboard/`: dashboards, referencias de interfaz y configuración histórica de despliegue.
- `artifacts/`: artefactos históricos asociados a las herramientas archivadas.
- `requirements.txt`: dependencias compartidas del entorno histórico, conservadas como referencia; no garantizan que estos scripts sean reproducibles hoy.

Las rutas originales de datos, algunos archivos de entrada/salida y ciertos servicios externos ya no están disponibles o pueden haber cambiado. Los documentos históricos describen el contexto en el que se desarrolló este material.
