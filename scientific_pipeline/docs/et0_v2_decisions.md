# ET₀ V2 — registro de decisiones

Las decisiones de Dirección 00 sustituyen las propuestas anteriores de la PR #65; la historia permanece en Git. [Revisión metodológica](et0_methodology_review_v2.md) · [Presentación](assets/et0_v2_experiment_protocol.pptx).

## Autorizado

- Objetivo: comparar fórmulas, AHC, ML y ANN por estación, con iguales entradas, contra PM_FAO56_V1 calculada; precisión, sesgo y cobertura con pocos sensores. No es validación instrumental independiente.
- DEVELOPMENT hasta 31/12/2019; evaluación final retrospectiva 2020–2025, no prospectiva independiente.
- Entradas: HS-Ra Tmin/Tmax/Ra; HG-Rs media de extremos/Rs; Valiantzas Tmin/Tmax/RHmedia/Ra. Anual sin calendario añadido; estacional/mensual con indicadores equivalentes al AHC, sin variables ocultas.
- AHC anual, estacional y mensual ajustados directamente por estación y fórmula hasta 2019; publicar todas las granularidades sin elegir por TEST. Ajuste adicional anual hasta 2025: exclusivamente dentro de muestra y separado.
- ML: validación por año omitido y media simple de RMSE anuales; conservar siete algoritmos y rejillas V1 cuando sean compatibles.
- ANN: [16], [32], [64], [64,32], [64,32,16]; tres semillas y parámetros generales V1. Comparar A: último 15 % cronológico, y E: 15 % aleatorio reproducible compartido entre candidatos; interna elige épocas, año omitido puntúa arquitectura y 2020–2025 evalúa configuraciones congeladas. No escoger regla ganadora con TEST.
- RMSE, MAE, bias, n y cobertura; cohorte común para error y propia para disponibilidad.

## Pendiente de contrato ejecutable — 05

Fijar soporte mínimo por estación/grupo/año, elegibilidad, desempates, asignaciones reproducibles A/E, detalles de compatibilidad de rejillas, trazabilidad y reglas de publicación antes de cualquier ejecución. No se inventan umbrales operativos.

## Quality y series cortas

IB07: histórico elegible como caso separado. IB09: revisión Quality antes de decidir admisión; conservar raw. IB101: inventariar soporte sin fabricar evaluación temporal.

## Futuro

Umbrales de uso y validación prospectiva requieren decisiones y datos independientes. El bloque retrospectivo no es un holdout globalmente prístino. No se ha ejecutado V2.
