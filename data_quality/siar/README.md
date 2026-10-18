# Informes de calidad SIAR

Esta carpeta queda reservada para los informes de Quality Audit v1:

- `siar_station_audit.csv`
- `siar_missing_days.csv`
- `siar_dataset_manifest.csv`

Sus columnas y reglas están definidas en [docs/siar/quality.md](../../docs/siar/quality.md). La situación de adquisición y el estado de certificación constan en [docs/siar/sync-status.md](../../docs/siar/sync-status.md).

**No hay todavía artefactos Quality versionados en esta carpeta.** Quality Audit PRE fue ejecutado externamente y preservado como evidencia; POST permanece pendiente. R1 es una etapa histórica superada y R2 está en ejecución autónoma. Esto no implica que la evidencia PRE no exista ni autoriza generar snapshots ficticios.

La incorporación de artefactos PRE/POST requiere un cambio explícito revisado; este documento no incorpora copias ni resultados. No añadir datos inventados ni modificar los CSV raw de `datos_siar_baleares/`. Esta carpeta solo recibirá salidas de una auditoría ejecutada y revisada.
