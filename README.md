# SIAR y evapotranspiración de referencia (ET₀)

Este repositorio reúne observaciones meteorológicas SIAR, investigación sobre estimación diaria de ET₀ y herramientas para explorar y preservar resultados. La comparación científica utiliza una **referencia Penman–Monteith FAO-56 calculada**, no una medición independiente mediante lisímetro.

## Empezar aquí

1. [Recorrido científico y cronológico](docs/RECORRIDO_CIENTIFICO.md): datos, Quality, dataset, fórmulas, calibración, ML y ANN, V1 y V2, desarrollo, freeze y retrospectiva.
2. [Memoria científica V2](scientific_pipeline/docs/ET0_V2_MEMORIA_CIENTIFICA.md): métodos, ecuaciones, resultados, bibliografía y límites para revisión.
3. [Auditoría científica independiente TFG–TFM–ET₀ V2](scientific_pipeline/docs/et0_v2_auditoria_cientifica_tfg_tfm.md): concordancias, discrepancias metodológicas y comprobaciones pendientes.
4. [Estado](docs/PROJECT_STATUS.md), [hoja de ruta](docs/ROADMAP.md) y [auditoría documental del 26/11](docs/AUDITORIA_DOCUMENTAL_2026-11-26.md).
5. [Datos y ubicaciones](docs/DATA_LOCATIONS.md) y [preservación y recuperación](docs/backup/README.md).

## Líneas operativas

- [SIAR Sync](siar-sync/README.md): adquisición observacional y publicación evolutiva. Sus CSV actuales no sustituyen el dataset científico congelado.
- [SIAR Forecast](forecast_collector/README.md) y [AEMET Forecast](aemet_forecast_collector/README.md): captura y preservación de previsiones, independientes de los experimentos retrospectivos.
- [Web científica V1](web_app/docs/release-v03.md) y [web V2 estable en LAN](web_app/docs/et0-v2-stable.md): actas y procedimientos, no comprobaciones en tiempo real.

V2 permanece **NOT_PUBLISHED**. Las pruebas retrospectivas no son validación prospectiva ni instrumental externa. Los estados de ejecución y copias deben contrastarse con sus responsables: un merge o un recibo histórico no acreditan el estado actual de un servicio.

Los capítulos históricos de la raíz conservan sus rutas y contenido; el [recorrido](docs/RECORRIDO_CIENTIFICO.md#documentos-históricos-de-la-raíz) explica su posición. Se conserva también el [índice anterior](docs/README_PROJECT_HISTORY_2026-11-25.md).
