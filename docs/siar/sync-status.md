# Estado de sincronización SIAR — Baleares

**Actualizado:** 2026-10-13\
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

En el snapshot PRE, los 2.346 huecos están registrados como `PENDING_RETRY`, `NOT_ATTEMPTED`, con `attempt_count=0`. El PRE no declaró ningún `CONFIRMED_NO_DATA`; Quality Recovery R1 está pausado y no se ha generado un POST.

Quality Recovery R1 = `PAUSED` en checkpoint 206/431 bloques; causa = `UNEXPLAINED_CHANGE` IB07; `POST = NOT GENERATED`. `FORENSIC-IB07-001` está autorizado y R1 permanece detenido hasta resolverlo.

## Hallazgo `UNEXPLAINED_CHANGE` IB07

IB07 tiene `Fecha_Baja` oficial **2012-11-13**. El HEAD previo terminaba en **2012-11-12** y ahora aparece una fila de **2012-11-13**. El origen de la escritura aún no está demostrado; no se declara la fila válida ni inválida. `FORENSIC-IB07-001` está autorizado. Quality Recovery R1 permanece detenido hasta resolverlo.

## Hallazgo IB09

El periodo oficial de IB09 es:

- Fecha_Instalacion: **2005-10-24**
- Fecha_Baja: **2018-10-10**

El CSV raw conserva una fila fechada **2005-10-23**, anterior al inicio oficial. Se conserva sin cambios y se cuenta fuera del periodo oficial. Por este hallazgo IB09 mantiene el resultado **FAIL**; no se ha corregido ni eliminado la observación.

## Catálogo de estaciones

El hallazgo **CATALOG-TZ** (snapshot local `estaciones_baleares.csv`: huso **31**; `Info/ESTACIONES`: huso **30**) está pendiente de investigación semántica y se clasifica como sistemático, pero no bloqueante por sí solo. No se corregirá automáticamente el catálogo ni se usará ese campo científicamente hasta conocer su significado. Por sí solo no degrada el resultado PRE ni bloquea la certificación, salvo que se demuestre que afecta al periodo o a la semántica de los datos.

## Saneamiento operativo de permisos posterior a v2.3.2

El modo esperado de los CSV canónicos es `0644`. La lógica correctora forma parte de SIAR Sync desde v2.3.2; el saneamiento puntual de archivos existentes fue operativo, posterior a v2.3.2, y no constituye una nueva versión funcional.

Resultado verificado:

- **12/12 CSV en modo `0644`**;
- **SHA-256 PRE = POST en 12/12**;
- contenido raw sin cambios.

No se reescribieron filas, esquema, fechas ni observaciones.

## Estado de adquisición

Los CSV canónicos presentes son IB01–IB11 e IB101. Quality Audit PRE permanece válido. Quality Recovery R1 sigue pausado en 206/431 bloques por `UNEXPLAINED_CHANGE` IB07; `POST = NOT GENERATED`. `FORENSIC-IB07-001` está autorizado y R1 no debe reanudarse hasta resolver el hallazgo.

Las políticas, reglas de clasificación y esquemas de informes están en [Quality Audit v1](quality.md). Este documento resume el estado comunicado; no genera ni contiene snapshots o informes de auditoría.
