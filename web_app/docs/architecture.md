# Contrato de arquitectura y datos WEB-V0.1

Fecha: 2026-10-24. Revisión inicial sobre main `35927e1ef07775c405fa6d8be9fe5957835d2f39`.
El mandato WEB-V0.1 autoriza un componente separado `web_app/`, sin cambiar contratos científicos ni servicios de adquisición.

## Separación de responsabilidades

```text
05 scientific_pipeline → artefactos aprobados CSV/JSON
                       → conversión web versionada → Parquet + metadata
                       → consultas DuckDB → FastAPI → React / ECharts
```

La conversión transforma representación y selecciona columnas para consulta. No vuelve a calcular ET₀, elegibilidad, Quality ni métricas científicas. Los archivos científicos se leen sin sobrescribirlos. Los contenedores de consulta montan los derivados en solo lectura; no reciben acceso de escritura a raw, Dataset v1 o Quality.

DuckDB encaja con 75.496 filas y consultas analíticas filtradas por estación/fecha sin un servidor de base de datos adicional. Parquet reduce transferencia y tiempo de lectura. FastAPI define una interfaz independiente del frontend; React y TypeScript permiten componentes reutilizables y ECharts cubre serie/scatter/barras con una sola librería principal. Docker Compose mantiene portable el despliegue doméstico.

## Autoridades e identidad

| Concepto | Identidad vigente |
|---|---|
| Dataset | `siar-balears-observational-v1` |
| Dataset SHA256 | `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb` |
| Filas preservadas | 75.496; 75.495 dentro del periodo oficial |
| Quality POST | `SIAR-QA-POST-20261020T193140Z` |
| Audit source repo SHA | `18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865` |
| Canonical source repo SHA | `2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9` |
| Pipeline de comparación | `a0b440fa3ab92e4593be521e4e819c6421327656` |
| Run de comparación | `run-116d6a7b955d4496947d75ada203436e` |
| CSV de cuatro métodos SHA256 | `f55e4ed12691b84dbca4e090c49da538d5e7fad4f2ef371e1307a649cfb1af44` |
| Métricas JSON SHA256 | `4c5890fac6cef86be21b4a4a8f0abec2fefc3c4d78f172f37681d176b904e8bd` |
| Manifest científico SHA256 | `78d249f3736ff0aeefbdd4599b4f2e77ded0c76c21bff26f274664e282ebee62` |

La fuente completa comprobada se encuentra en el workspace de 05, bajo `scientific-results/four-classical-20261024T042250Z/comparison/classical-baselines.csv`. `PROJECT-ARTIFACT-BACKUP-V1` está PASS para el snapshot remoto verificado declarado en main; un derivado web nuevo requiere su propia transferencia e ingestión explícitas y no amplía retroactivamente aquel snapshot.

El manifiesto derivado debe registrar versión del convertidor, SHA de cada fuente consumida, SHA de cada archivo generado, dataset ID/SHA, POST, ambos SHA audit/canonical, pipeline y run. Nunca presentar el SHA del Parquet como SHA del dataset científico JSON.

## Modelos y representación

| ID público | Columna CSV | Método en métricas | Sufijo de elegibilidad |
|---|---|---|---|
| PM_FAO56_V1 | ET0_PM_FAO56_V1 | ET0_FAO56 | PM |
| HS_RA_V1 | ET0_HS_RA_V1 | ET0_Hargreaves | HS |
| VALIANTZAS_V1 | ET0_VALIANTZAS_V1 | ET0_Valiantzas | VAL |
| HG_RS_V1 | ET0_HG_RS_V1 | ET0_Hargreaves_Rs | HGRS |

EtPMon es una serie de referencia independiente del registro de predictores; todos los valores ET₀ se muestran en mm/día. `eligible_*` significa que el método puede calcular; `evaluated_*` identifica pares admitidos frente a EtPMon. No se sustituyen estos booleanos por una máscara propia de nulos. Nulos siguen siendo nulos; cero es un dato válido, y predicciones negativas se conservan con sus flags. IB09 permanece presente con Quality FAIL y cero predicciones/evaluaciones autorizadas.

El catálogo completo contiene 12 estaciones: nueve ACTIVE y tres CLOSED (IB07, IB09, IB101). Los periodos oficiales proceden de POST/metadatos aprobados; mínimo/máximo observado son cobertura, no una redefinición del periodo oficial. `Huso` tiene hallazgo CATALOG-TZ-001 abierto; no se usa para resolver coordenadas científicas.

## Métricas aprobadas y filtros

`metrics.json` es la autoridad. La API filtra sus filas, conserva los números publicados y devuelve el ámbito explícito. No promedia MAE/RMSE/R² de estaciones ni reconstruye métricas para intervalos arbitrarios.

`common` significa exclusivamente la cohorte publicada `PM_HS_VAL_HGRS`, n global=70.860. La selección visual de dos modelos no redefine esa cohorte como un nuevo par. `own` conserva la población propia de cada método. La semántica anterior `paired` sigue correspondiendo a PM/HS y no se renombra como comparación de cuatro.

| Solicitud | Grupo fuente | Semántica |
|---|---|---|
| global sin estación | global / all | Todo el periodo publicado y todas las estaciones |
| global con estación | station / IBxx | Todo el periodo publicado de esa estación |
| year sin estación | year / YYYY | Todas las estaciones, por año |
| year con estación | station_year / IBxx:YYYY | Estación y año publicados |
| month sin estación | month / MM | Mes calendario agrupado sobre todos los años y estaciones |
| season sin estación | season / DJF,MAM,JJA,SON | Temporadas boreales agrupadas sobre todos los años y estaciones |
| month/season con estación | No existe | Declarar no disponible/rechazar; no ignorar el filtro ni recalcular |

Los grupos adicionales publicados son `season_year`; no existen `station_month` ni `station_season`. El frontend debe hacer visible este límite. El rango de fechas de serie/scatter no se aplica a las métricas, que informan `period_filter_applied=false`. Las temporadas siguen el contrato científico: diciembre pertenece al año siguiente de DJF.

Error firmado=predicción−EtPMon. MAE/RMSE/bias en mm/día; R² predictivo=1−SSE/SST. R² negativo es válido; indefinido se presenta como ausencia, nunca como cero. La API no altera los ceros de referencia ni las cohortes para conseguir un mejor valor.

## API v0.1

| Ruta | Contrato funcional |
|---|---|
| GET /api/health | Estado de servicio y disponibilidad de derivados verificados; un proceso vivo sin datos válidos no acredita readiness. |
| GET /api/stations | Catálogo, estado ACTIVE/CLOSED, nombre, periodo, observaciones y Quality. |
| GET /api/models | Registro metadata-driven con ID, etiqueta, color, referencia e inputs. |
| GET /api/dataset/info | Procedencia, hashes, versión de conversión, run y limitaciones. |
| GET /api/timeseries | Estación y rango de fechas validados, lista de modelos permitidos; respuesta acotada con método de reducción y recuentos explícitos. |
| GET /api/metrics | Selección publicada por estación/modelo/grupo y own/common; alcance y periodo independientes del rango visual. |
| GET /api/scatter | Pares evaluables del modelo, fecha/estación y cohorte solicitados; muestra acotada con N total y método de muestreo. |
| GET /api/coverage | Recuentos por estación de observaciones, Quality y elegibilidad/evaluación por modelo. |

Fechas ISO civiles, sin inventar zona horaria del dato diario. Identificadores/modelos/grupos se validan contra listas permitidas. SQL parametrizado para valores; nombres de columnas únicamente desde metadata controlada. Ningún endpoint acepta rutas arbitrarias ni ofrece escritura o recálculo científico.

Series y scatter tienen límites duros de puntos/respuesta. Una agregación o muestra se etiqueta como visualización y comunica filas originales y puntos devueltos. Nunca se calculan métricas sobre la muestra del scatter. Las ausencias en el calendario no deben mostrarse como continuidad meteorológica comprobada.

## Limitaciones que debe mostrar la interfaz

- EtPMon es referencia calculada SIAR; no es medición independiente de lisímetro/ground truth.
- IB09 está excluida de cálculo/evaluación por Quality FAIL, con observaciones preservadas.
- IB101 tiene un registro corto.
- UNITS_PROVENANCE_PARTIAL; instrumentación por estación/periodo no exhaustivamente verificada y viento 2 m por convención SIAR.
- Métricas own y common representan poblaciones distintas; no hay vencedor universal.
- Forecast, AHC y ML todavía no están integrados; no inferir D+n ni reutilizar coeficientes legacy.
- Cobertura de cálculo/evaluación no equivale a completitud de auditoría ni a cobertura observacional del calendario.

## Operación y evolución

Despliegue inicial LAN, sin NAT ni exposición pública. Inspeccionar host/usuario, checkout, Docker, espacio y puertos antes de elegir directorios y puerto. Producción utiliza checkout propio y datos derivados separados, con reinicio y healthchecks; no reutilizar el checkout residual SIAR Sync sin análisis. Código por rama `06/...`, PR, CI y merge antes de desplegar.

Los futuros modelos entran mediante metadata y artefactos aprobados. AHC temporal y su contrato de visualización ya están PASS/FROZEN, pero quedan fuera del alcance deliberado de WEB-V0.1 y se incorporarán como WEB-V0.2 sin recalcular ciencia. ML necesita productores verificables; forecast necesita issue time, valid time y horizontes científicos cerrados. Añadir columnas o servicios de visualización no autoriza esas decisiones científicas.

## Lectura de referencia

Se revisaron README, PROJECT_STATUS, ROADMAP, MILESTONES, DATA_LOCATIONS, las bitácoras históricas 1.2/1.3, todos los documentos de `scientific_pipeline/docs/` y la evidencia oficial de resultados PM/HS, tres y cuatro baselines. Los avisos históricos de bloqueo en unidades/espera de dataset están superados por las decisiones posteriores fechadas; no se transforman en nuevos bloqueos web.
