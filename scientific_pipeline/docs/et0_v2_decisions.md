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

## Contrato ejecutable — 05

El [contrato de preparación V2](et0_v2_preparation.md) fija soporte por estación, grupo y año, elegibilidad, desempates, asignaciones reproducibles A/E, rejillas y trazabilidad. Los controles de soporte son técnicos. Los umbrales operativos siguen pendientes.

## Quality y series cortas

IB07 e IB101 se inventarían por separado como series históricas. El 11/11/2026 Dirección 00 aprobó la excepción científica de IB09 para DEVELOPMENT de V2 entre 2005-10-24 y 2018-10-10, ambos inclusive. Se conserva el FAIL de Quality V1, la verificación oficial independiente UNRESOLVED y la fila del 2005-10-23 fuera de alcance. La [decisión preservada](../../data_quality/siar/ib09-handoff-v2-development/decision.md) exige detener la admisión ante evidencia oficial contradictoria o fallo de reconciliación.

## Futuro

Umbrales de uso y validación prospectiva requieren decisiones y datos independientes. El bloque retrospectivo ya formó parte de experimentos anteriores. V2 está en preparación, con lanzamiento pendiente de autorización.
