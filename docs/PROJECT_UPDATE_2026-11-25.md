# Actualización consolidada — 25/11/2026

Base auditada: [main en b000be0](https://github.com/lependens/Capitulo-1/commit/b000be05ca36345a5e973612b8c17ea3822c0a06). El corte previo se conserva [íntegro](PROJECT_UPDATE_2026-11-16.md). El [inventario](PROJECT_STATUS.md) concentra los estados y responsables; la [hoja de ruta](ROADMAP.md) concentra las prioridades.

## Cambios comprobados en GitHub

| PR | Fusión (hora de Madrid) | Alcance |
|---|---|---|
| [#75](https://github.com/lependens/Capitulo-1/pull/75) | 16/11 17:24 · [`35667d1`](https://github.com/lependens/Capitulo-1/commit/35667d14d0bfeb1dd3690745d9926ce6ba4be473) | Consolidación anterior |
| [#76](https://github.com/lependens/Capitulo-1/pull/76) | 24/11 17:24 · [`7529ca8`](https://github.com/lependens/Capitulo-1/commit/7529ca8397cb8fb4589c0146bdd9fd1c23b9cda1) | Retrospectiva, contratos y certificación técnica |
| [#78](https://github.com/lependens/Capitulo-1/pull/78) | 24/11 22:33 · [`077b75f`](https://github.com/lependens/Capitulo-1/commit/077b75fa994fd7e90f6673dd977ff12fb764b551) | Memoria para revisión independiente |
| [#77](https://github.com/lependens/Capitulo-1/pull/77) | 25/11 10:26 · [`b000be0`](https://github.com/lependens/Capitulo-1/commit/b000be05ca36345a5e973612b8c17ea3822c0a06) | Web de revisión y recuperación V0.3 |

La [comparación con el corte anterior](https://github.com/lependens/Capitulo-1/compare/35667d14d0bfeb1dd3690745d9926ce6ba4be473...b000be05ca36345a5e973612b8c17ea3822c0a06) contiene 88 commits, 34 de publicación automática SIAR. La revisión de PR registró 78 entradas (#1–#78), 75 fusionadas y tres abiertas (#72–#74, AEMET). La consulta de ramas devolvió 77 nombres, principalmente ramas de entregas conservadas. No figuran releases de GitHub; las versiones se identifican por actas, commits y artefactos.

## Adquisición y preservación

- **SIAR Sync V3.** La rama principal ya contiene [13 CSV evolutivos](../datos_siar_v3), incluido AL01 junto a las estaciones baleares. Los commits automáticos prueban publicación en GitHub. Los partes de desarrollo describen SQLite, cursor y ledger de huecos, planificador, escritura atómica y reconciliación con la publicación separada. La versión exacta desplegada, la raíz activa, la cola y el acceso LAN requieren confirmación de 01. El paquete preservado del 14/11 acredita su propio alcance según [el parte anterior](PROJECT_UPDATE_2026-11-16.md).
- **SIAR Forecast.** El último parte de piloto superó el gate histórico que esperaba la primera captura automática con backup. Para acreditar continuidad, 02 debe aportar últimas emisiones y cobertura, y 02B los recibos del reconciliador y alcance del restore.
- **AEMET Forecast.** A1 tiene persistencia OAuth y restauración comunicadas. A2 publicó un run COMPLETE y preservó también el PARTIAL previo, pero el reconciliador devolvió exit 1 por un run ya copiado. A3 sigue condicionado a cerrar A2. 02 debe aportar prueba de idempotencia, commit del runtime y evidencia de la programación cuando corresponda. Véanse [#72](https://github.com/lependens/Capitulo-1/pull/72), [#73](https://github.com/lependens/Capitulo-1/pull/73) y [#74](https://github.com/lependens/Capitulo-1/pull/74).
- **Preservación.** El sistema general de snapshots y los backups de emisiones tienen coberturas distintas. Aplicar los [procedimientos de verificación y restore](backup/README.md) y confirmar con 02B la cobertura de cada nueva ruta. Una copia verificada de un paquete no acredita automáticamente el conjunto del servidor.

## Ciencia y exploradores

La retrospectiva V2 usa modelos congelados sobre 2020–2025. El expediente documenta 108 celdas AHC y 105 por cada fase ML, ANN A y ANN E, 4.935 checkpoints trazados y 9.612 filas de métricas contrastadas. La prueba independiente de inferencia cubre tres modelos representativos. Siguen documentados los límites de R², salidas negativas, procedencia del notebook, cobertura y AHC dentro de muestra. Consultar [certificación](../scientific_pipeline/docs/et0_v2_certificacion_evidencia.md), [contrato de inferencia](../scientific_pipeline/docs/et0_v2_contrato_inferencia.md) y [memoria](../scientific_pipeline/docs/ET0_V2_MEMORIA_CIENTIFICA.md). La revisión científica y bibliográfica sigue pendiente con 05 y Dirección.

La [guía de integración](../web_app/docs/et0-v2-integration.md) documenta catálogo VERIFIED en V2 y recuperación aislada de V0.3. La recuperación comprobó 18 hashes y consultas representativas. El merge de #77 muestra checks backend y frontend correctos. La web científica V2 sigue en revisión LAN con publicación NOT_PUBLISHED. Dirección cerró la fase 1 y mantiene ambas webs disponibles para comparación. La continuidad después de las comprobaciones del 25/11 queda por confirmar con 06.

## Hallazgos de organización

El repositorio separa adquisición observada, dos colectores, Quality, pipeline científico, web y documentación de backup. La auditoría de estructura abarcó 599 entradas del árbol. El README anterior, el estado y la hoja de ruta mezclaban cierres de octubre con instrucciones vigentes. Se da una función distinta a cada índice y se mantienen los cortes históricos. Las guías de instalación específicas siguen en su componente y las de restauración en backup y en las actas de release.

En una revisión futura, conviene convertir los bloques fechados de `DATA_LOCATIONS.md` en un índice con anexos históricos y separar la guía DRAFT de la guía LAN VERIFIED dentro de la documentación web. Revisar con 01 y 05 si los nombres repetidos de CSV en `datos_siar_baleares/` y `datos_siar_v3/` requieren un manifiesto de procedencia; son líneas de datos con propósitos distintos. La limpieza de ramas debe decidirse después de cotejar su contenido y preservación.
