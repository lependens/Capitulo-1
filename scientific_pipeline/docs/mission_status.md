# ET0-BASELINES-PREP — estado de integración

## Estado vigente — 24/10/2026

**VALIANTZAS-V1-AND-CLASSICAL-BASELINES-EXPORT = PASS**.
[Resultados oficiales de tres métodos y CSV](../results/classical-20261024T034529Z/README.md).
PR #33; pipeline 4ac52e25c803cb2301316c5cb1353bf30cdb427a; 68 tests y CI 3.11/3.12/3.13 PASS.
VAL 70.944 predicciones / 70.878 evaluables / 0 fallos; Dataset v1 intacto.

## Snapshot anterior — 23/10/2026 (histórico)

**Dataset Observational v1 construido/validado PASS. Primera ejecución oficial PM/HS PASS.** La espera de POST y handoff descrita en el registro de preparación inferior es HISTÓRICA / SUPERADA.

- POST vigente: SIAR-QA-POST-20261020T193140Z; publicación canónica 2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9; auditado 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865.
- Dataset SHA256: 0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb.
- PR #30: tolerancia VPD explícita 1e-12 kPa absoluta / 0 relativa, sin cambios en dataset o Quality.
- PR #31: PM-VPD-FLOAT-TOLERANCE-AND-BASELINES PASS; 62 tests y CI 3.11/3.12/3.13 PASS según evidencia publicada.
- PM: 70.896 calculadas / 70.861 evaluadas / 0 fallos. HS: 70.979 / 70.912 / 0 fallos. Paired n=70.861.
- Mantener limitaciones de referencia SIAR, procedencia parcial, viento convencional, exclusión IB09 por FAIL y muestra IB101.

[Informe oficial, métricas y artefactos](../results/pm-vpd-20261023T194553Z/README.md). No se modifican ni regeneran resultados en esta actualización documental. Valiantzas permanece LEGACY / FUTURE BASELINE REVIEW; los siguientes gates metodológicos constan en [ROADMAP](../../docs/ROADMAP.md).

## Registro histórico de preparación — 19/10/2026

El texto siguiente conserva la evidencia y las condiciones de aquella entrega. Las menciones de dataset/ejecución pendientes se refieren al estado del 19/10, superado por el resultado vigente anterior.


Fecha: 2026-10-19 (Europe/Madrid). Decisión de Dirección 00: cerrar forense y avanzar preparación integrada.

## Estado de preparación al 19/10/2026

- Forense: **RESOLVED_FOR_V1**.
- Interpretación aceptada: Radiacion MJ m⁻² día⁻¹; VelViento m/s; lectura directa, sin ×0.0864 ni /3.6.
- Viento: convención SIAR de 2 m aceptada para v1. La falta de historial instrumental exhaustivo no es STOP.
- Limitaciones no bloqueantes: **UNITS_PROVENANCE_PARTIAL** y **station_period_provenance_not_exhaustively_verified**, conservadas en trazabilidad/manifiestos.
- Evidencia positiva de altura distinta de 2 m: detener PM en el ámbito afectado hasta resolución explícita; no bloquea automáticamente HS.
- **ET0-BASELINES-PREP = READY_FOR_DATASET_V1**. Integración, pruebas, revisión y CI científica verificadas; la construcción del dataset oficial permanece pendiente de POST y handoff.

Esta decisión supera los diagnósticos preliminares BLOCKED_UNITS y WIND_HEIGHT_UNVERIFIED de la fase anterior. No reescribe las cifras forenses ni transforma una convención aceptada en verificación instrumental por estación/periodo. No se continuarán búsquedas históricas para avanzar esta misión.

## Alcance preservado

La investigación partió de main `6f1dd95c415730ec29843055d3a32a7ac5a636ad` y documentación 1.2, seguida de código, outputs legacy y documentación 1.3. La auditoría y el contraste offline quedan como evidencia histórica. El informe forense generado conserva sus valores y diagnóstico fechado; una nota posterior de decisión 00 define la interpretación operativa actual.

No se ha ejecutado el Dataset Observational v1 definitivo ni publicado resultados finales de baselines sobre él. No se modifica adquisición, Quality, raw, CSV canónicos, legacy o el esquema canónico de 23 columnas. Las pruebas sintéticas no sustituyen Quality POST.

## Preparación integrada entregada

La entrega incorpora CLI `python -m siar_science` con validación, construcción desde handoff, PM, HS y evaluación; salidas nuevas con predicciones, métricas y manifiesto. Se han comprobado:

1. Funciones FAO-56/HS con ejemplos independientes, dominios y políticas numéricas registradas.
2. Adaptador del handoff `SIAR-QUALITY-CLOSEOUT`, esquemas POST oficiales y hashes de fuentes e informes, sin fabricar cierre autorizado.
3. QC por método, preservación de observaciones y flags, referencia SIAR separada de predicción.
4. Ejecución sintética de extremo a extremo y pruebas de rechazo de contradicciones, duplicados y sobrescritura.
5. Trazabilidad de dataset/configuración/código/cohortes/outputs en manifiesto y CI científica verificable.

El [README](../README.md), [contrato](contracts.md) y [supuestos](assumptions.md) documentan el comportamiento implementado.

## Verificación de cierre

- Main inicial: `6f1dd95c415730ec29843055d3a32a7ac5a636ad`.
- Implementación validada: `7a1ece4c9082a15973baa1cbb70f7373a21c63db`.
- Suite científica local: **48 tests PASS**, incluidos ejemplos FAO publicados, QC/eligibilidad, métricas, handoff sintético con hashes reales, catálogo, pipeline y CLI.
- CI científica: **PASS en Python 3.11/3.12/3.13**, paquete instalado y entry point probado; [ejecución verificada](https://github.com/lependens/Capitulo-1/actions/runs/39173293712).
- Suite forense preservada: 9 tests PASS tras actualizar exclusivamente la interpretación documental del cierre de 00; no se añadieron cálculos históricos.
- Revisión independiente: corregidas coherencia de cohortes pareadas, protección raw y trazabilidad del productor de predicciones; regresiones cubiertas por los tests de integración.
- Diff-check completo contra main: PASS. Alcance: scientific_pipeline y su workflow CI; sin cambios en raw, legacy, catálogo original, Quality, Sync o Forecast.
- [PR #23](https://github.com/lependens/Capitulo-1/pull/23) conserva el registro canónico de revisión, CI y SHA de merge; no se introduce un SHA de merge autorreferencial en sus propios archivos.

Los commits de cierre documental posteriores no cambian las fórmulas ni los resultados de las pruebas de la implementación identificada; la PR vuelve a verificar la CI sobre su head antes de fusionarse.

## Decisiones científicas congeladas para v1

PM diario FAO-56 y HS sin calibración usan media de extremos. PM fija G=0, γ estándar, albedo=0.23 y VPD de extremos; Rnl no tiene floor inferior por defecto. Conservar negativos con flags, sin sustituciones silenciosas ni ±3σ global. HS usa 0.0023·0.408 fijo y Ra. EtPMon sigue como referencia calculada SIAR, no verdad independiente.

Métricas: n, MAE, RMSE, MBE/bias, R² predictivo; cohortes propias y pareadas separadas, referencias cero conservadas. Futura calibración/ML exige cortes temporales globales y aprendizaje solo en TRAIN; no forma parte de esta misión.

## Gate de ejecución definitiva

Preparar y verificar el software no autoriza generar el dataset final ahora. La construcción definitiva espera el handoff autorizado de 00 después de Quality closeout: `gate=SIAR-QUALITY-CLOSEOUT`, informes POST originales, catálogo/metadatos, hashes, unidades y fuentes identificadas. El mismo gate sigue vigente aunque la forense esté resuelta.

La capa está READY_FOR_DATASET_V1. La disponibilidad de Quality POST y Dataset v1 es un estado separado; la misión no los declara completados ni ejecuta los baselines definitivos para anticiparlos. Ready for Quality POST: **YES** como software preparado para recibir el handoff; generación y ejecución oficial: **NO, esperando handoff de 00**.
