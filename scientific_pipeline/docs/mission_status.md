# ET0-BASELINES-PREP — retorno por condición STOP

Fecha: 2026-10-18 (Europe/Madrid).

Estado: **BLOCKED_UNITS**, no READY_FOR_DATASET_V1.

## Base y alcance

- Main inicial comprobado: `6f1dd95c415730ec29843055d3a32a7ac5a636ad`.
- Consulta inicial de PR abiertas: ninguna.
- Merge SHA final: no existe; no se solicita fusionar una preparación incompleta.
- No se ha generado Dataset Observational v1, ejecutado baselines sobre los CSV
  canónicos, ni alterado adquisición, Quality, raw o legacy.

## Bloqueo demostrado

El [manual oficial API SIAR v2.2, mayo de 2025](https://servicio.mapa.gob.es/siarweb/documentos/manuales/ManualTecnico_WEB_API_SIAR.pdf),
página 14, documenta alternativas de unidad para Radiacion diaria dependiendo de
la configuración. La descripción de VelViento diaria no concreta su unidad.
Página 13 permite unidades alternativas para el viento horario, pero no demuestra
qué unidad emplean los registros diarios de Baleares.

En el commit base, `siar-sync/siar_worker.py:635-642` solicita Datos diarios con
token, Id, fechas y DatosCalculados; no fija ni registra una configuración de
unidades. `docs_1.2` afirma m/s y MJ/m², pero no aporta una evidencia de la
configuración de la cuenta/origen vinculada al snapshot adquirido. Una escala
numérica plausible o la proximidad de una fórmula a EtPMon no constituye esa
evidencia.

Esto **no demuestra que los CSV tengan unidades incorrectas**, sino que no se
puede certificar la interpretación científica a partir de la trazabilidad
disponible. Corresponde a la condición STOP «unidad SIAR crítica no demostrable».
No requiere modificar el esquema canónico ni reinterpretarlo por conjetura.

## Handoff necesario para continuar

Dirección 00 debe aportar o coordinar evidencia oficial aplicable a los datos
adquiridos: unidad de Radiacion y VelViento diarios, configuración efectiva de
la cuenta/servicio y alcance temporal y por estación; confirmar también unidad
de EtPMon para la evaluación. Guardar referencia y hash del documento/export o
respuesta verificable y vincularlos a los hashes de los snapshots. Si existen
cambios de unidad históricos, delimitarlos antes de diseñar la transformación
derivada. Para PM verificar altura efectiva del viento por estación/periodo;
una descripción genérica del trípode no sustituye el metadato de cada sensor.

No se pide permiso para asumir unidades. Se requiere evidencia para resolver la
decisión científica. No contactar a terceros ni cambiar SIAR Sync como parte de
esta preparación.

## Trabajo conservado y pendiente

Auditoría documental y de resultados históricos en `legacy_audit.md`; contrato
propuesto y controles previos en `contracts.md`. Ambos son preparatorios y no
certifican Dataset v1. Las discrepancias legacy explicables se documentan sin
adoptarlas como norma nueva.

Se conservan borradores locales de funciones matemáticas, métricas y elegibilidad;
no constituyen un pipeline integrado ni una entrega científica operativa. Falta
integración de validación, construcción bloqueada por handoff/POST, CLI, manifiesto,
pruebas de extremo a extremo y CI científica. Esta PR documental no publica esos
borradores de código.

Verificación local preparatoria: 14 tests matemáticos independientes PASS con
ejemplos publicados FAO-56 (incluidos PM y HS); solo verifican los borradores
locales, no las unidades de SIAR ni una integración de producción. No hay CI
científica creada/ejecutada para esos borradores. La comprobación del diff de
la entrega documental debe pasar antes de publicar la PR.

Ready for Quality POST: **NO como capa ejecutable**. El contrato propuesto sirve
para coordinar el handoff, sin cambiar el Quality Audit ni producir artefactos
POST. Tras resolver el bloqueo, completar integración y tests antes de declarar
READY_FOR_DATASET_V1; seguir esperando el handoff de 00 antes de generar Dataset v1.
