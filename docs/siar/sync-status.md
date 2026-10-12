# Estado de sincronización SIAR — Baleares

**Actualizado:** 2026-10-12  
**Versión de SIAR Sync en producción:** v2.3.3  
**Estado del backfill:** temporalmente alcanzado para las 12 estaciones

## Estado declarado

Dirección 00 declara alcanzado el backfill temporal de adquisición para las 12 estaciones de Baleares. El inventario actual contiene estos CSV canónicos:

IB01, IB02, IB03, IB04, IB05, IB06, IB07, IB08, IB09, IB10, IB101 e IB11.

La ejecución actual de SIAR Sync ha terminado por ahora. **No se ha lanzado una segunda pasada** como parte de Quality Audit v1.

Este estado describe el objetivo temporal comunicado, no una certificación científica. La presencia de un CSV no demuestra que todas las fechas esperadas tengan registros válidos, que no haya duplicados o anomalías, ni que cualquier día sin fila corresponda a una ausencia confirmada de datos en SIAR.

## Permisos heredados y saneamiento operativo

El modo canónico esperado para cada *_datos_completos.csv es 0644. La lógica correctora de SIAR Sync ya forma parte de v2.3.2. La normalización puntual de archivos creados previamente es un **saneamiento operativo posterior a v2.3.2**, no una nueva versión funcional.

Inventario PRE comunicado para los 12 CSV:

| Modo observado | Estaciones |
|---|---|
| 0644 | IB01, IB02, IB04, IB10, IB101, IB11 |
| 0600 | IB03, IB05, IB06, IB07, IB08, IB09 |

La operación efectiva queda pendiente de coordinación separada con 01/02. No se realiza desde esta PR ni se reescriben los CSV. Procedimiento requerido: calcular SHA-256 PRE; cambiar únicamente el modo a 0644; calcular SHA-256 POST; comprobar igualdad PRE/POST; verificar que los 12 modos terminan en 0644. No deben cambiar contenido, filas, esquema, fechas ni observaciones.

Una auditoría PRE puede registrar los modos reales encontrados. Si la normalización precede a la auditoría definitiva, registrar el timestamp verificable en permission_normalized_at_utc. GitHub no conserva los bits ordinarios 0600/0644 y no puede ejecutar esta normalización del filesystem.

## Pendiente

- auditar hashes, estructura, fechas y modo de los 12 CSV;
- revisar ventanas oficiales a partir del snapshot reproducible estaciones_baleares.csv, contrastándolo con la autoridad Info/ESTACIONES antes de certificar;
- usar un cutoff explícito para cada estación activa;
- limitar la ventana de una estación cerrada a Fecha_Baja;
- clasificar huecos persistentes con evidencia, separando ausencia confirmada, error de consulta, respuesta inválida, cuota y falta de consulta;
- emitir los tres informes definidos en [Quality Audit v1](quality.md) tras ejecutar y revisar la auditoría;
- coordinar y verificar el saneamiento operativo de permisos;
- certificar científicamente calidad y cobertura.

Hasta completar y revisar esos pasos, el backfill se considera temporalmente alcanzado y la certificación de calidad permanece pendiente.
