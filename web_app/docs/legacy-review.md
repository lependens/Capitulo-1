# Revisión de la aplicación histórica

Fecha: 2026-10-24. Base inspeccionada: `35927e1ef07775c405fa6d8be9fe5957835d2f39`.
La aplicación histórica es una referencia funcional; sus datos, filtros y resultados no son autoridad científica v1.

## Fuentes inspeccionadas

- `legacy/dashboard/app_dash.py`: comparación de modelos, selector de estación y años, serie temporal, scatter, diferencia mensual y tablas local/global.
- `legacy/dashboard/app_dash_online.py`: variante para Gunicorn/Render, carga global de CSV y reconstrucción de diferencias dentro de la app.
- `legacy/dashboard/procfile`: `web: gunicorn app_dash:server`.
- `legacy/dashboard/DASH.MD`: descripción histórica de una variante ANN. El documento menciona `app_dash_v2.py`, que no está en este directorio; no se supone que sus prestaciones estén implementadas en los archivos preservados.
- `legacy/dashboard/mapa_interactivo.html`: referencia histórica de mapa.
- `docs_1.2_Tratamiento de datos y cálculo de estimaciones.md`, `docs_1.3_Análisis errores.md` y `scientific_pipeline/docs/legacy_audit.md`: diferencias metodológicas y actualizaciones que superan los resultados históricos.

## Disposición funcional

| Función legacy | Decisión | Aplicación en WEB-V0.1 |
|---|---|---|
| Selector de estación | Rediseñar | Catálogo completo de 12 códigos, incluidos IB101 y estaciones CLOSED; nombre, periodo y Quality visibles. |
| Selección de años | Rediseñar | Fechas explícitas para series y scatter; las métricas publicadas conservan su periodo completo identificado. |
| Serie temporal de varios métodos | Conservar la idea | Gráfico grande con modelos desde metadata, referencia EtPMon separada, zoom, tooltip y reducción de puntos solo visual. |
| Scatter y línea 1:1 | Conservar la idea | Predicción frente a EtPMon; pares autorizados por `evaluated_*`; muestreo visual explícito y tamaño total. |
| Barras de diferencia mensual | Rediseñar | Mostrar bias publicado por mes y cohorte; sin reconstruir diferencias desde medias de cohortes distintas. |
| Tabla de errores por estación y global | Conservar la idea | MAE, RMSE, bias, R² y N publicados; own/common y ámbito visibles. |
| Explicación de entradas de modelos | Conservar la idea | Metadata con identificadores versionados, requisitos de inputs y referencia científica. |
| Detección dinámica de modelos ANN | Conservar como principio futuro | Registro metadata-driven; no habilitar ANN ni reinterpretar columnas históricas como modelos aprobados. |
| Mapa de estaciones | Posponer | Incorporar tras estabilizar la base; no derivar coordenadas desde UTM/Huso sin resolver CATALOG-TZ-001. |
| Selector de estación de origen AHC | Descartar en v0.1 | Era un placeholder sin efecto sobre los CSV. AHC requiere misión científica y contrato temporal previos. |
| PM como referencia perfecta | Descartar | EtPMon es la referencia calculada SIAR; PM es otro baseline, con errores publicados distintos de cero. |
| Sunshine / PM cielo claro | Descartar en v0.1 | No forma parte de los cuatro baselines oficiales autorizados. |
| Recálculo de métricas en callbacks | Descartar | API consume métricas científicas aprobadas; navegador solo presenta resultados. |
| `dropna` global y eliminación por variables auxiliares | Descartar | Conservar nulos y elegibilidad propia de cada método. |
| Filtros ±3σ y limpieza aprendida en toda la serie | Descartar | No forman parte del contrato vigente; no alterar población científica para mejorar un gráfico. |
| AHC calculado/evaluado sobre toda la serie | Descartar | Histórico full-series/in-sample; no prueba generalización y no se convierte en AHC_V1. |
| Bootstrap/CDN externo y arquitectura Dash/Gunicorn | Rediseñar | React/TypeScript/Vite, una librería ECharts, API FastAPI y dos servicios Docker portables. |

## Riesgos científicos observados

`app_dash_online.py` elimina registros sin ET0_calc, TempMedia o Radiacion, calcula diferencias y métricas en la capa web, y asigna error cero al PM usado como referencia. `app_dash.py` usa `dropna()` sobre todos los modelos para la serie y las métricas. Ninguno de esos comportamientos define las cohortes v1.

La auditoría científica identifica además resultados históricos con metadatos placeholder IB03/IB04, distinción RMSE/RRMSE y Pearson²/R², exclusión histórica de referencias cero y AHC dentro de muestra. Las actualizaciones de 2026-10-24 definen VALIANTZAS_V1 y HG_RS_V1 distintos del legado. No se ejecutó código legacy ni se modificó su contenido.

Las ideas retenidas son selección de estación, comparación temporal, scatter, evolución mensual y tablas de errores. Se reconstruyen sobre artefactos aprobados, con su procedencia y sin importar la lógica científica antigua.
