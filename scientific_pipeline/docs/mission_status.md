# ET0-BASELINES-PREP — estado tras UNITS-OFFLINE-FORENSIC

Fecha: 2026-10-18 (Europe/Madrid).

Estado refinado: **WIND_HEIGHT_UNVERIFIED** para PM, **PROVENANCE_PENDING** para
certificación. Interpretación de unidades resuelta con evidencia offline sólida.
No READY_FOR_DATASET_V1: preparación integrada todavía incompleta.

## Base y alcance

- Main inicial comprobado: `6f1dd95c415730ec29843055d3a32a7ac5a636ad`.
- Consulta inicial de PR abiertas: ninguna.
- Merge SHA final: no existe; no se solicita fusionar una preparación incompleta.
- No se ha generado Dataset Observational v1 ni sus resultados finales, ni
  alterado adquisición, Quality, raw o legacy. El refinamiento de 00 autoriza
  expresamente el diagnóstico offline de hipótesis sobre snapshots preservados.

## Conclusión refinada: interpretación, certificación y altura

El [manual oficial API SIAR v2.2, mayo de 2025](https://servicio.mapa.gob.es/siarweb/documentos/manuales/ManualTecnico_WEB_API_SIAR.pdf),
página 14, documenta alternativas de unidad para Radiacion diaria dependiendo de
la configuración. La descripción de VelViento diaria no concreta su unidad.
Página 13 permite unidades alternativas para el viento horario, pero no demuestra
qué unidad emplean los registros diarios de Baleares.

Sin embargo, la auditoría inicial dio demasiado peso a esa limitación del
manual antes de agotar la evidencia offline. `docs_1.1` documenta las unidades
de adquisición, `docs_1.2` el método, y PM/PM2/calculadora consumen directamente
MJ m⁻² día⁻¹ y m/s. No se encuentra conversión histórica observacional.

La misión UNITS-OFFLINE-FORENSIC contrasta las cuatro combinaciones de escalas
en 22 archivos preservados y dos kernels PM, conservando todos los demás inputs.
En los 12 canónicos (75.236 pares, 2004–2025), la lectura directa tiene RMSE
0.087799 mm/día en PM2 y 0.022715 en sensibilidad FAO-56; las alternativas
se separan ampliamente. Se estratifica por estación, año y condiciones, incluida
una excepción local de 26 días de IB101:2008 que demuestra por qué un mínimo
de error aislado no decide la unidad. No se ajusta la fórmula contra EtPMon.

**Unit interpretation:** Radiacion MJ m⁻² día⁻¹; VelViento m/s, sólidamente
respaldados para los snapshots auditados. No aplicar ×0.0864 ni /3.6.
**Provenance certification:** todavía no hay certificado histórico completo
de configuración de origen; conservar esa limitación y los hashes.
**Wind sensor height:** no acreditada por estación/periodo. Todos los cálculos
del experimento suponen 2 m; no demuestran altura efectiva del anemómetro.

Se retira BLOCKED_UNITS como ambigüedad de escala para estos snapshots.
Permanece el bloqueo PM específico WIND_HEIGHT_UNVERIFIED. La procedencia
pendiente se registra aparte y no convierte otra vez la interpretación en
desconocida. No se cambia esquema ni Quality.

## Handoff necesario para continuar

Para levantar el bloqueo PM falta ficha instrumental o declaración oficial de
alcance suficiente sobre altura efectiva del viento y su vigencia por estación.
No confundir altitud del catálogo con altura de sensor ni tratar un comentario
de programa como ficha instrumental. Completar procedencia en el handoff POST
con referencias verificables y hashes, conservando explícita la evidencia
offline de interpretación y su alcance.

No se requiere una consulta adicional SIAR para elegir entre las dos escalas
contrastadas. En esta misión no se ha consultado API ni web adicional, contactado
a terceros ni cambiado SIAR Sync. PR #23 permanece DRAFT y no se fusiona.

## Trabajo conservado y pendiente

Auditoría histórica en `legacy_audit.md`, búsqueda textual/binaria offline en
`units_offline_evidence.md`, contraste y conclusión en `units_offline_forensic.md`.
Contrato actualizado en `contracts.md`. Herramientas, tests, 4.256 filas agregadas
de métricas, ejemplo IB04 y manifest con hashes en `../forensics/`. Ninguno de
ellos es Dataset v1 ni certificación POST.

Se conservan borradores locales de funciones matemáticas, métricas y elegibilidad;
no constituyen un pipeline integrado ni una entrega científica operativa. Falta
integración de validación, construcción bloqueada por handoff/POST, CLI, manifiesto,
pruebas de extremo a extremo y CI científica. La PR publica exclusivamente la
auditoría y la herramienta forense aislada, no esos borradores del pipeline.

Verificación local preparatoria: 14 tests matemáticos independientes PASS con
ejemplos publicados FAO-56 (incluidos PM y HS); solo verifican los borradores
locales, no las unidades de SIAR ni una integración de producción. No hay CI
científica creada/ejecutada para esos borradores. La comprobación del diff de
la entrega debe pasar antes de actualizar la PR. Suite forense independiente:
9 tests PASS, incluyendo ejemplo FAO publicado, ejemplo IB04, máscaras,
ceros, métricas, cohortes pareadas, hashes de fuentes/resultados y barrera para
no trasladar una conclusión narrativa a nuevos snapshots sin revisión. No hay CI
integrada del pipeline; la verificación forense no la sustituye.

Ready for Quality POST: **NO como capa ejecutable**. El contrato propuesto sirve
para coordinar el handoff, sin cambiar el Quality Audit ni producir artefactos
POST. Tras resolver el bloqueo, completar integración y tests antes de declarar
READY_FOR_DATASET_V1; seguir esperando el handoff de 00 antes de generar Dataset v1.
