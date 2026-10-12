# Estado de sincronización SIAR — Baleares

**Actualizado:** 2026-10-12  
**Versión SIAR Sync en producción:** v2.3.3
**Backfill temporal:** alcanzado para las 12 estaciones
**Quality Audit v1 PRE:** COMPLETADO

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

Los 2.346 huecos están registrados como `PENDING_RETRY`, `NOT_ATTEMPTED`, con `attempt_count=0`. **No se ha declarado ningún `CONFIRMED_NO_DATA`**: no hubo consultas de recuperación que probaran una respuesta SIAR válida con cero registros.

Quality Recovery / POST está autorizado, pero todavía no está completado. La certificación observacional permanece abierta hasta revisar el POST y resolver los hallazgos pendientes.

## Hallazgo IB09

El periodo oficial de IB09 es:

- Fecha_Instalacion: **2005-10-24**
- Fecha_Baja: **2018-10-10**

El CSV raw conserva una fila fechada **2005-10-23**, anterior al inicio oficial. Se conserva sin cambios y se cuenta fuera del periodo oficial. Por este hallazgo IB09 mantiene el resultado **FAIL**; no se ha corregido ni eliminado la observación.

## Catálogo de estaciones

El snapshot local reproducible `estaciones_baleares.csv` indica huso **31**, mientras que `Info/ESTACIONES` indica huso **30**. Es una discrepancia pendiente de decisión y contraste. No se corregirá automáticamente el catálogo ni se dará por certificada la referencia afectada hasta resolverla.

## Saneamiento operativo de permisos posterior a v2.3.2

El modo esperado de los CSV canónicos es `0644`. La lógica correctora forma parte de SIAR Sync desde v2.3.2; el saneamiento puntual de archivos existentes fue operativo, posterior a v2.3.2, y no constituye una nueva versión funcional.

Resultado verificado:

- **12/12 CSV en modo `0644`**;
- **SHA-256 PRE = POST en 12/12**;
- contenido raw sin cambios.

No se reescribieron filas, esquema, fechas ni observaciones.

## Estado de adquisición

Los CSV canónicos presentes son IB01–IB11 e IB101. La ejecución actual de SIAR Sync terminó por ahora y no se ha lanzado una segunda pasada. El backfill temporal alcanzado no equivale a certificación científica.

Las políticas, reglas de clasificación y esquemas de informes están en [Quality Audit v1](quality.md). Este documento resume el estado comunicado; no genera ni contiene snapshots o informes de auditoría.
