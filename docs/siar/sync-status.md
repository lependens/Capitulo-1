# Estado de sincronización SIAR — Baleares

**Actualizado:** 2026-10-18\
**Versión SIAR Sync en producción:** v2.3.3
**Backfill temporal:** alcanzado para las 12 estaciones
**Quality Audit v1 PRE:** COMPLETADO — válido

## Resultado de Quality Audit v1 PRE

| Métrica | Resultado |
|---|---:|
| Estaciones auditadas | 12 |
| ACTIVE | 9 |
| CLOSED | 3 |
| WARN | 11 |
| FAIL | 1 (IB09) |
| INCOMPLETE | 0 |
| Días esperados | 77.840 |
| Días con dato | 75.494 |
| Huecos PRE | 2.346 |
| Cobertura ponderada | 96,99 % |

En el snapshot PRE, los 2.346 huecos quedaron registrados como `PENDING_RETRY`, `NOT_ATTEMPTED`, con `attempt_count=0`. El PRE no declaró ningún `CONFIRMED_NO_DATA`; esos estados son inmutables y no describen necesariamente la recuperación actual.

R1 queda superado como etapa histórica. **R2 está en ejecución autónoma; POST permanece pendiente.** Un `INVALID_RESPONSE` aislado se registra y trata dentro del ámbito afectado, sin bloquear globalmente R2. El cierre Quality requiere terminar la recuperación y revisar POST; esta actualización no genera artefactos ni certifica todavía el dataset.

## Antecedente histórico R1 — IB07

El antecedente documentado de R1 fue: IB07 con `Fecha_Baja` oficial **2012-11-13**, HEAD previo hasta **2012-11-12** y una fila nueva de **2012-11-13**. El checkpoint 206/431 y `FORENSIC-IB07-001` pertenecen al historial de R1; no constituyen una pausa vigente de R2. Este resumen no infiere un nuevo dictamen científico sobre esa fila.

## Hallazgo IB09

El periodo oficial de IB09 es:

- Fecha_Instalacion: **2005-10-24**
- Fecha_Baja: **2018-10-10**

El PRE detectó una fila raw fechada **2005-10-23**, anterior al inicio oficial; la conservó sin cambios y la contó fuera del periodo. Por ello IB09 tuvo resultado **FAIL en PRE**. Es un hallazgo del snapshot histórico; todavía no se dispone del resultado POST.

## Catálogo de estaciones

El hallazgo **CATALOG-TZ** (snapshot local `estaciones_baleares.csv`: huso **31**; `Info/ESTACIONES`: huso **30**) está pendiente de investigación semántica y se clasifica como sistemático, pero no bloqueante por sí solo. No se corregirá automáticamente el catálogo ni se usará ese campo científicamente hasta conocer su significado. Por sí solo no degrada el resultado PRE ni bloquea la certificación, salvo que se demuestre que afecta al periodo o a la semántica de los datos.

## Saneamiento operativo de permisos posterior a v2.3.2

El modo esperado de los CSV canónicos es `0644`. La lógica correctora forma parte de SIAR Sync desde v2.3.2; el saneamiento puntual de archivos existentes fue operativo, posterior a v2.3.2, y no constituye una nueva versión funcional.

Resultado verificado:

- **12/12 CSV en modo `0644`**;
- **SHA-256 PRE = POST en 12/12**;
- contenido raw sin cambios.

Estas verificaciones corresponden al saneamiento puntual de permisos: no se reescribieron filas, esquema, fechas ni observaciones durante esa operación. No describen los posibles cambios de contenido producidos posteriormente por Recovery.

## Estado de adquisición

Los CSV canónicos presentes son IB01–IB11 e IB101. PRE está completo, R1 es histórico, R2 avanza autónomamente y POST sigue pendiente. El siguiente hito es Quality closeout y, después, Dataset Observacional v1; la captura prospectiva de forecasts avanza en paralelo.

Las políticas, reglas de clasificación y esquemas de informes están en [Quality Audit v1](quality.md). Este documento resume el estado comunicado; no genera ni contiene snapshots o informes de auditoría.
