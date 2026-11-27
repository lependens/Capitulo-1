> **Actualización posterior, 27/11/2026:** la [PR #87](https://github.com/lependens/Capitulo-1/pull/87) incorporó la [auditoría científica TFG–TFM–ET₀ V2](../scientific_pipeline/docs/et0_v2_auditoria_cientifica_tfg_tfm.md). Es una revisión independiente focalizada con comprobaciones pendientes, que complementa esta auditoría documental fechada. V2 continúa NOT_PUBLISHED.

# Auditoría documental — 26/11/2026

Base de lectura: [main 258c833](https://github.com/lependens/Capitulo-1/commit/258c83320fae00dc8994c643c5af50fdd46a1846). El árbol contiene 545 rutas versionadas en este corte. La actividad automática puede mover main después. Se revisaron la estructura, índices y documentos científicos y operativos de referencia; esta guía no certifica cada afirmación histórica ni inspecciona servicios, copias o secretos.

## Evidencia GitHub

- [#79](https://github.com/lependens/Capitulo-1/pull/79) fusionó el corte documental anterior. [#80](https://github.com/lependens/Capitulo-1/pull/80) amplió memoria e inventario experimental; [#81](https://github.com/lependens/Capitulo-1/pull/81) integró la fase 2 web; [#82](https://github.com/lependens/Capitulo-1/pull/82) entregó metadatos DEVELOPMENT. Las cuatro estaban fusionadas al consultar GitHub.
- [#83](https://github.com/lependens/Capitulo-1/pull/83), [#84](https://github.com/lependens/Capitulo-1/pull/84) y [#85](https://github.com/lependens/Capitulo-1/pull/85) estaban abiertas en borrador. #85 se dirige a la rama de #84; ninguna se debe tratar como código integrado en main. Consultar sus fichas para el estado posterior.
- Los commits automáticos muestran publicaciones en GitHub, no acreditan por sí solos el estado del servicio, cobertura o continuidad.

## Partes operativos y pendientes

La [web estable](../web_app/docs/et0-v2-stable.md) registra despliegue y restore aislado el 25/11; la [PR #81](https://github.com/lependens/Capitulo-1/pull/81) describe verificaciones de aquella entrega. No se han comprobado ahora los servicios. [#83](https://github.com/lependens/Capitulo-1/pull/83) informa de capturas SIAR sin filas y causa no determinada; no declara adquisición útil reciente. [#84](https://github.com/lependens/Capitulo-1/pull/84) registra una captura tras reintento por timeout, con identidad de emisión e integridad pendientes; el reintento 429 se había probado en simulación. [#85](https://github.com/lependens/Capitulo-1/pull/85) describe una captura controlada parcial y deja pendiente el primer ciclo natural ampliado. La descripción de #85 aún solicita adaptación web; #83 documenta después la adaptación para ocho estaciones. Son evidencias fechadas de PR, no mediciones actuales.

- **02/02W:** contrastar ejecución, emisiones útiles, integración, CI y pendientes de ambos colectores y la interfaz conjunta con sus responsables.
- **02BW/02B:** contrastar configuración Windows, causa de la ventana y cobertura efectiva de preservación. La [prueba Windows](backup/windows-ingestion.md) acredita un paquete sintético, no todos los archivos del ordenador. La aplicación visual solicitada y la eliminación de la ventana requieren su propio trabajo y verificación.
- **Capitulo-2:** línea nueva comunicada en el contexto del proyecto; no se infiere estado de su repositorio ni cobertura de backup.
- **Ciencia:** revisión científica y bibliográfica independiente pendiente; V2 sigue NOT_PUBLISHED. No confundir metadatos DEVELOPMENT ni AHC in-sample con resultados TEST de candidatos descartados.

## Organización y limitaciones

El README es portada y el [recorrido](RECORRIDO_CIENTIFICO.md) enlaza fuentes en orden científico. Los siete capítulos de raíz conservan contenido y rutas; se añadió un aviso de lectura histórica. No se alteraron datos, código, modelos ni servicios. Los documentos fechados mantienen sus afirmaciones en contexto; la cabecera de estado y hoja de ruta remite a este corte. No se han movido archivos históricos, evitando romper enlaces externos. Queda pendiente una revisión completa de afirmaciones y enlaces externos, y la conciliación de cambios posteriores en main y de los responsables operativos.

## Segunda revisión: inventario, enlaces y PR abiertas

Se enumeraron los 91 Markdown versionados de esta rama (incluidos dos nuevos), con búsquedas sistemáticas de estados, y se revisaron los índices y fuentes principales. Un comprobador automático de enlaces Markdown encontró 646 destinos relativos existentes y 20 anclas reconocidas, sin errores detectados. No analiza todas las variantes de Markdown, HTML ni enlaces construidos dinámicamente. Se consultaron 162 URL externas únicas: 145 respondieron 200 a HEAD, cuatro 404/410, nueve 403, dos errores de conexión/certificado y dos LAN excluidas. Uno de los 404 era un DOI truncado por el extractor; Kaggle y el PDF UIB tienen páginas indexadas aunque HEAD falló. Por ello las respuestas negativas no se clasifican como roturas confirmadas. Los notebooks privados pueden requerir autenticación. No se han sustituido fuentes científicas sin identificar una alternativa equivalente.

GitHub mostraba abiertas y en borrador #72, #73 y #74, además de #83–#86. [#84](https://github.com/lependens/Capitulo-1/pull/84) declara consolidar #72–#74; no se ha demostrado equivalencia completa ni decidido cerrar las originales. Su rama estaba nueve commits detrás de main; [#85](https://github.com/lependens/Capitulo-1/pull/85) tiene como base la rama de #84. GitHub no mostraba revisiones ni revisores solicitados en las PR consultadas. Todas figuran bajo la cuenta `lependens`; las responsabilidades siguientes son las funciones del proyecto identificadas en su contexto, no asignaciones formales de GitHub.

| PR | Responsable funcional orientativo y siguiente paso | CI de la cabeza consultada |
|---|---|---|
| [#72–#74](https://github.com/lependens/Capitulo-1/pull/72) | 02W AEMET y 02B: cotejar su contenido con #84. Dirección decide si se cierran como sustituidas o se integran; no duplicar cambios. | Éxito individual histórico en las tres; no acredita #84. |
| [#83](https://github.com/lependens/Capitulo-1/pull/83) | 02W SIAR/02: revisar integración y evidencia de capturas útiles; Josep revisa la interfaz y Dirección decide integración. No afirmar recuperación SIAR con capturas vacías. | Éxito en `1276fb2`. |
| [#84](https://github.com/lependens/Capitulo-1/pull/84) | 02W AEMET/02B: resolver o explicar el fallo CI de HTTP 503, comprobar identidad e integridad del ciclo y reconciliar con main; después revisión y decisión de Dirección. | Fallo en `2cad584`, [ejecución](https://github.com/lependens/Capitulo-1/actions/runs/45821718166). |
| [#85](https://github.com/lependens/Capitulo-1/pull/85) | 02W AEMET/02: resolver CI, comprobar primer ciclo natural ampliado y su preservación; coordinar integración con #84 y #83. | Fallo en `ba666c8`, [ejecución](https://github.com/lependens/Capitulo-1/actions/runs/45845989871). |
| [#86](https://github.com/lependens/Capitulo-1/pull/86) | Auditoría documental y revisión por Josep/Dirección; revisar sus límites y aprobar expresamente antes de fusionar. | No apareció una ejecución asociada a la cabeza consultada. |

Los logs de CI confirman 108 pruebas superadas y una fallida en #84, y 109 superadas y dos fallidas en #85. El fallo compartido esperaba conservar `payload.raw` tras HTTP 503; #85 añade una discrepancia sobre conservación de staging. Es necesario analizar código, contrato y logs antes de decidir si falla implementación o prueba. La primera captura natural ampliada estaba prevista para las 19:20 del 26/11; no se afirma su resultado sin nueva evidencia. El chat 02BW comunicó una consulta de la tarea Windows con `WindowStyle Hidden` y último resultado 0 a las 17:31: eso no identifica la causa de la ventana ni acredita copia remota.

### Alcance restante

La auditoría mecánica no demuestra que cada afirmación, fórmula, cita o resultado sea correcto. La revisión académica independiente, los enlaces que exigen acceso y el estado en directo de los servicios siguen pendientes. Las actas antiguas son fuentes fechadas; los cambios posteriores deben contrastarse antes de actualizar conclusiones.