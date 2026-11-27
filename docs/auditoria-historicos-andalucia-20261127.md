# Auditoría inicial de históricos manuales de Andalucía

Fecha de revisión: 2026-11-27. Estado: EN CURSO; sin autorización científica para importar a V3 a partir de este documento.

## Procedencia
Archivos aportados por el propietario, recopilados manualmente desde SIAR para el TFM. Origen local: carpeta estacions andalucia, estructurada por provincias. Se observa diversidad de formatos; no debe inferirse el código de estación exclusivamente del nombre de archivo. Ejemplo: Almería/a1.txt tiene cabeceras AL10. La sección inicial de Almería/GENERAL ALMERIA.csv identifica AL02.

Fuente de contraste operativa: /home/josep/siar-sync-v3-data/csv/AL02_datos_completos.csv, leída el 27/11/2026. Es una fuente derivada SIAR V3, no una comprobación independiente de la web SIAR.

## Muestra preliminar AL02
Se inspeccionaron las cinco fechas 01–05/06/2020, en la sección inicial del CSV histórico y en AL02 de V3. Las temperaturas, humedad, viento, dirección, radiación, precipitación y horas observadas son compatibles en estas filas con el redondeo visible del archivo histórico; esto no equivale a una auditoría automatizada completa de cada campo.

| Fecha | ET0 histórica | EtPMon V3 | Diferencia histórica menos V3 |
|---|---:|---:|---:|
| 01/06/2020 | 5.18 | 5.195760 | -0.015760 |
| 02/06/2020 | 4.61 | 4.607870 | +0.002130 |
| 03/06/2020 | 5.26 | 5.240662 | +0.019338 |
| 04/06/2020 | 5.48 | 5.445802 | +0.034198 |
| 05/06/2020 | 5.42 | 5.406935 | +0.013065 |

Las discrepancias ET0 no se explican todas por redondear directamente EtPMon a dos decimales. Causa no determinada: no corregir ni sustituir valores sin investigación; comprobar definiciones, fuente original y posible revisión o recálculo.

## Pendiente
- Inventario y hashes de cada fuente; identificar cada archivo y las secciones con código de estación y evidencia, especialmente archivos sin cabecera.
- Comparación reproducible por fecha y variable, con reglas explícitas de redondeo y tratamiento de valores ausentes.
- Contrastar cinco fechas por estación con V3 cuando exista solapamiento y, si no existe, con el portal oficial SIAR. Registrar respuesta y evidencia.
- Resolver discrepancias y determinar compatibilidad con las 23 columnas; las variables ausentes no deben inventarse.
- Evaluar duplicados, periodos, huecos, conflictos y cómo planificador y SQLite reconocen datos importados.
- Preparar prueba aislada y copias de seguridad antes de cualquier incorporación. No se han modificado los CSV, SQLite ni trabajos de producción en esta revisión.
