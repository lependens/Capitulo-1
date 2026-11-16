# Unidades observacionales SIAR: evidencia forense offline

> **Evidencia histórica preservada — actualización 23/10/2026.** Este informe describe la investigación offline y sus decisiones fechadas; no es el estado operativo actual del dataset. Las esperas de POST/ejecución pertenecen a esa etapa. Dataset v1 y la ejecución oficial PM/HS ya están PASS, con [resultados posteriores](../results/pm-vpd-20261023T194553Z/README.md). No se reescriben cifras forenses ni resultados legacy.

Fecha: 2026-10-18. Árbol inspeccionado: `87d7cf5d00c8ca587e5208e2ec529455f7abc2c6` más archivos locales de trabajo. No se realizó ninguna consulta de red o API, no se ejecutaron adquisidores, no se alteraron observaciones, metadatos, código ni resultados. Este documento registra evidencia textual y estructural; el contraste numérico de hipótesis se entrega separadamente.

## Decisión vigente 00 — 2026-10-19

La investigación forense se cierra **RESOLVED_FOR_V1**. Se acepta Radiacion en MJ m⁻² día⁻¹ y VelViento en m/s, sin ×0.0864 ni /3.6, y uso directo del viento con **convención SIAR de 2 m**. No se exige continuar búsquedas históricas para habilitar v1. `UNITS_PROVENANCE_PARTIAL` y `station_period_provenance_not_exhaustively_verified` se mantienen como limitaciones de trazabilidad no bloqueantes. Solo una evidencia positiva de altura distinta de 2 m reabre ese punto y detiene el cálculo PM afectado hasta resolverlo.

Las secciones que siguen conservan la evidencia y sus límites epistemológicos a fecha 2026-10-18; la ausencia de certificación instrumental exhaustiva que documentan **no es un STOP vigente**. Esta decisión no convierte las convenciones aceptadas en verificación instrumental por estación ni reescribe cifras o resultados forenses. Véanse [supuestos](assumptions.md) y [estado operativo](mission_status.md).

## Conclusión de evidencia offline (2026-10-18)

**Interpretación histórica explícita:** el proyecto identifica `VelViento` y `VelVientoMax` como m/s y `Radiacion` como MJ/m² en la documentación de adquisición. El cálculo diario legacy interpreta esta última como MJ m⁻² día⁻¹ y el viento como u₂ a 2 m. No son inferencias basadas solo en rangos de valores: existen declaraciones literales y usos directos en código.

**Procedencia no cerrada:** esas declaraciones locales describen la intención y el contrato esperado del proyecto. No se encontró una captura de configuración efectiva del proveedor, respuesta con unidades incorporadas o historial de cambios que vincule inequívocamente ese contrato con cada lote, estación y periodo del CSV canónico. La ausencia de conversiones en los adquisidores conserva la escala recibida; no identifica por sí misma cuál era la escala recibida.

**Altura del viento no acreditada por estación/periodo:** la indicación «2 m» aparece en fórmulas, comentarios y una calculadora genérica; no en fichas instrumentales o campos del catálogo. Altitud de estación y altura del anemómetro son magnitudes distintas. El repositorio no permite elevar automáticamente `VelViento` a observación verificada a 2 m.

La evidencia offline refuerza la hipótesis m/s y MJ m⁻² día⁻¹ como interpretación histórica del dataset, pero no demuestra configuración efectiva ni altura de sensor. Las comparaciones contra EtPMon podrán contrastar coherencia, sin sustituir esas pruebas de procedencia y metadatos.

## Cobertura de la búsqueda

Se inventariaron los 243 archivos seguidos por Git disponibles al inicio y los archivos locales visibles, incluidos ocultos y extensiones sin nombre convencional. Se buscó en documentación, adquisición antigua, SIAR Sync, validación de calidad, ET₀, ML histórico, dashboards, configuraciones, plantillas HTML y colectores forecast. Los borradores nuevos de `scientific_pipeline/` se identificaron como productos de esta misión, no como evidencia independiente anterior.

Se buscaron literalmente y con variantes: `0.0864`, `0,0864`, `86400`, `1000000`, `1e6`, `1e+6`, conversiones con `3.6`, unidades W/m², MJ/m², m/s, km/h, watt/vatio, anemómetro, altura/height, sensor y unidad/unit. Se inspeccionaron las dos hojas de cálculo mediante ZIP/XML, sin abrir Excel ni ejecutar contenido activo. No existen notebooks `.ipynb` ni PDF en el árbol local inspeccionado. Las imágenes y modelos binarios no constituyen diccionarios de unidades por sí solos; los CSV preservados contienen nombres de campo sin unidades en sus cabeceras.

Una búsqueda adicional en el historial Git local detectó coincidencias numéricas en dependencias antiguamente versionadas —por ejemplo 86400 en expiración de caché—, ajenas a conversión meteorológica. No se usan como prueba de normalización SIAR. El inventario local no pretende acreditar configuraciones externas no guardadas en este repositorio.

## Declaraciones explícitas anteriores a esta misión

| Evidencia | Localización precisa | Qué demuestra y qué no |
|---|---|---|
| Tabla de salida de `siarIBconsulta2_corregido.py`: VelViento=m/s | `docs_1.1_Obtención de datos.md:90` | Declaración documental del campo del CSV de observaciones; no una captura de configuración efectiva de API. |
| VelVientoMax=m/s | `docs_1.1_Obtención de datos.md:91` | Misma declaración para la magnitud máxima; no habilita sustituir viento medio por máximo. |
| Radiacion=MJ/m² | `docs_1.1_Obtención de datos.md:97` | Declara energía por superficie para una salida diaria. La tabla no explica integración, huso temporal ni configuración de origen. |
| Salida diaria con viento m/s y radiación MJ/m² | `docs_1.2_Tratamiento de datos y cálculo de estimaciones.md:44` | Repite el contrato esperado de adquisición. |
| Rn y G diarios en MJ/m²/día | Mismo documento, líneas 88–89 | Unidades metodológicas de PM; no prueba de la unidad de Rs recibida. |
| u₂ definido como VelViento a 2 m, m/s | Mismo documento, línea 91 | Supuesto explícito al asociar campo y fórmula; no evidencia instrumental por estación. |
| Ra en MJ/m²/día | Mismo documento, línea 120 | Unidad de radiación extraterrestre calculada; no debe confundirse con unidad de radiación observada. |
| Viento leído directamente, comentario a 2 m y m/s | `legacy/et0/PM:63` | Uso e interpretación del modelo histórico. |
| Radiación leída directamente, comentario MJ/m²/día | `legacy/et0/PM:64` | Uso de `Radiacion` como energía diaria, sin conversión W→MJ. |
| Asignaciones U2 y Rs directas | `legacy/et0/PM2.py:155`, `legacy/et0/PM2.py:156` | El programa no transforma unidades o altura antes de PM. |
| Rs leído con comentario MJ/m²/d | `legacy/et0/et0alternativo2.py:28`, `legacy/et0/et0alternativo2.py:29` | Misma interpretación para Hargreaves con Rs medida. |
| Calculadora: viento m/s a 2 m; Rs MJ/m²/day | `legacy/et0/PM2EXCEL.py:35`, `legacy/et0/PM2EXCEL.py:36` | Especificación de entradas genéricas; no registros meteorológicos. |

`docs/ROADMAP.md:393` mantiene la tarea «verificar unidades». Los requisitos de unidades documentadas en sus líneas 298 y 404 son objetivos, no evidencias de resolución.

## Qué hacen realmente los adquisidores

### Adquisición histórica

`legacy/acquisition/siarIBconsulta2_corregido.py:9` define el endpoint antiguo y su línea 10 solicita `Diarios`. La URL de la línea 17 contiene Id, fechas y clave; no selecciona unidad ni adjunta configuración instrumental.

Las líneas 25 y 29 añaden estación y, si falta, fecha. La línea 111 aplica `pd.json_normalize`, la 125 concatena con datos existentes y la 137 escribe CSV. En ese recorrido no aparece conversión numérica de `VelViento` o `Radiacion`, ni columnas de unidad, altura o configuración efectiva. La conservación directa de valores hace especialmente importante identificar el contrato de origen.

### SIAR Sync actual

`siar-sync/siar_worker.py:12` define la API actual y las líneas 13–14, `Diarios/ESTACION`. La solicitud de las líneas 634–643 usa token, Id, FechaInicial, FechaFinal y `DatosCalculados=true`; no incluye unidad o altura. `DatosCalculados=true` no equivale a una declaración de unidades.

`_normalize_api_dataframe`, líneas 720–745, copia registros, normaliza el alias `humedadMin`, transforma Fecha, rellena columnas ausentes y ordena el esquema. No normaliza las magnitudes físicas. El esquema canónico de las líneas 34–59 tiene los campos meteorológicos, pero no campos para unidad/configuración/altura o sus periodos de vigencia.

`siar-sync/.env.example:12` introduce parámetros de descarga; el archivo completo no define parámetros para elegir unidades de radiación, viento o altura. La existencia de credenciales por entorno no demuestra qué configuración del proveedor se aplicó a ellas. No se leyeron valores secretos ni se consultaron cuentas.

Los módulos `data_quality/` validan integridad/cobertura según su alcance documentado; no aportan un diccionario de unidades del proveedor o historial de alturas de sensor. Sus manifiestos de calidad no deben interpretarse como certificación física implícita.

## Conversiones: resultado negativo contextualizado

No se encontró en los archivos de código/documentación anteriores a esta misión una conversión observacional con `0.0864`, `86400/1e6`, `86400`, `1000000`, `1e6` o `1e+6`. Tampoco una normalización km/h→m/s mediante división por 3.6. La única coincidencia textual pertinente con 3600 en el código de proyecto fue `legacy/et0/PM2.py:33`, conversión de segundos de coordenada DMS a grados, ajena a radiación y viento.

El factor **0.408 sí existe**, por ejemplo `legacy/et0/PM2.py:102` y `legacy/et0/et0alternativo2.py:24,30`. Convierte el término energético a equivalente de evaporación dentro de la fórmula y no convierte una radiación recibida en W/m² a energía diaria. No prueba que la API entregase una u otra unidad.

La ausencia de un factor no descarta una escala aplicada por el proveedor o una transformación anterior no preservada; sí descarta atribuir esa conversión a los recorridos de adquisición revisados.

## Hojas de cálculo y catálogo

`legacy/artifacts/ET0_Calculator.xlsx`, hoja `Inputs` (`xl/worksheets/sheet1.xml`):

- `A12`: Wind Speed (U2); `C12`: m/s; `D12`: Wind speed at 2m height.
- `A13`: Solar Radiation (Rs); `C13`: MJ/m²/day; `D13`: Measured solar radiation (if available). La cadena almacenada presenta corrupción del símbolo de superíndice en el XML inspeccionado, sin afectar la identificación de MJ y day.
- Son etiquetas de una plantilla de cálculo. No contienen identificación de sensor o vigencia por estación y no certifican sus datos de entrada.

`legacy/data/siar-derived/metadata/estaciones_listado.xlsx`, hoja `Sheet1`, celdas `A1:K1`: Estacion, Codigo, Termino, Longitud, Latitud, Altitud, XUTM, YUTM, Huso, Fecha_Instalacion, Fecha_Baja. No hay campo para altura de viento o unidad meteorológica. La búsqueda en sus celdas produjo una coincidencia numérica incidental en una coordenada UTM, no evidencia de conversión. Ninguno de los dos XLSX contiene partes VBA.

`estaciones_baleares.csv:1` y `legacy/data/siar-derived/metadata/estaciones_baleares.csv:1` preservan ese mismo esquema sin altura de anemómetro. `legacy/acquisition/siarIDIB.py:9,12,22` lee la hoja, filtra códigos IB y la exporta; no incorpora metadatos instrumentales adicionales.

## Evidencias excluidas de inferencia observacional

- La reproducción numérica de PM2 y sus variantes demuestra qué operaciones generan esos outputs históricos. No transforma la suposición de entrada de esos programas en un hecho sobre la API.
- Valores plausibles, correlaciones, ranking de errores y cercanía a EtPMon son contraste físico/numérico, con posible dependencia de un cálculo de referencia que usa entradas relacionadas. No sustituyen trazabilidad de unidad o altura.
- `forecast_collector/` y `aemet_forecast_collector/` se revisaron en la búsqueda textual, pero sus contratos/capturas pertenecen a adquisiciones independientes. No se traslada de ellos una unidad a `datos_siar_baleares/`.
- Los contratos y fórmulas de `scientific_pipeline/` fijan unidades normalizadas deseadas. Al haberse redactado en esta misión, no prueban el contrato histórico del proveedor ni sus instrumentos.

## Evidencia adicional que mejoraría procedencia (no bloqueante para v1)

Debe distinguirse la interpretación física más respaldada por evidencia offline de la evidencia administrativa/instrumental aún ausente. Para completar la certificación de procedencia sería necesario un artefacto verificable que vincule unidades y agregación diaria con la adquisición efectiva y su vigencia. Para completar la verificación instrumental exhaustiva de altura sería necesaria ficha o metadato instrumental por estación/periodo, o una declaración de alcance suficiente del proveedor. No se rellena esa ausencia suponiendo que altitud es altura del viento ni porque una función use la variable u₂.
