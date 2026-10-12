# Estado de sincronización SIAR — Baleares

**Actualizado:** 2026-10-12  
**Versión de SIAR Sync en producción:** v2.3.3

## Estado declarado

Dirección 00 declara alcanzado el backfill temporal de adquisición para las 12 estaciones de Baleares. El inventario actual del repositorio contiene estos CSV canónicos:

`IB01`, `IB02`, `IB03`, `IB04`, `IB05`, `IB06`, `IB07`, `IB08`, `IB09`, `IB10`, `IB101` e `IB11`.

La ejecución actual de SIAR Sync ha terminado por ahora. **No se ha lanzado una segunda pasada** como parte de Quality Audit v1.

Este estado describe el objetivo temporal comunicado, no una certificación científica. La presencia de un CSV no demuestra que todas las fechas esperadas tengan registros válidos, que no haya duplicados o anomalías, ni que cualquier día sin fila corresponda a una ausencia confirmada de datos en SIAR.

## Pendiente

- auditar hashes, estructura y fechas de los 12 CSV;
- revisar ventanas esperadas a partir de `estaciones_baleares.csv`;
- usar un cutoff explícito para cada estación activa;
- limitar la ventana de una estación cerrada a `Fecha_Baja`;
- clasificar cada hueco con evidencia, separando ausencia confirmada, error de consulta, respuesta inválida y falta de consulta;
- emitir los tres informes definidos en [Quality Audit v1](quality.md) tras ejecutar y revisar la auditoría;
- certificar científicamente calidad y cobertura.

Hasta completar y revisar esos pasos, el backfill se considera temporalmente alcanzado y la certificación de calidad permanece pendiente.
