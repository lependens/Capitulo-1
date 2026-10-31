# SIAR ET₀ Lab — WEB-V0.2-AHC

Aplicación web de consulta de Dataset Observational v1, los cuatro baselines oficiales y los resultados AHC publicados. El componente es de solo lectura: no calcula ET₀, coeficientes ni métricas y no escribe sobre raw, Quality, Dataset v1 o artefactos científicos.

## Arquitectura

```text
CSV + metrics.json + manifest.json aprobados
  → convertidor verificable
  → Parquet + metadatos derivados
  → DuckDB / FastAPI
  → React / ECharts
```

WEB-V0.2 añade una segunda conversión verificable desde las tablas congeladas de `visualization_contract`: cambia CSV a Parquet para consulta, pero conserva literalmente predicciones, métricas, coeficientes, cohortes y splits.

El navegador recibe únicamente las filas filtradas o muestras visuales acotadas. Las métricas proceden literalmente de `metrics.json`; no se calculan sobre el downsampling ni sobre el scatter.

## Crear el paquete web derivado

La conversión exige el CSV oficial de 75.496 filas, sus métricas y manifiesto juntos. Verifica los SHA publicados, el estado del run, el catálogo y el audit Quality antes de escribir un directorio nuevo.

```powershell
python -m web_app.backend.convert `
  --source C:\ruta\scientific-results\four-classical-20261024T042250Z\comparison `
  --output C:\ruta\web-derived-v01-20261024
```

El paquete registra `source_sha256`, `derived_sha256`, versión de conversión, Dataset ID/SHA, POST, pipeline, run y hashes de todos los derivados. El Parquet es un artefacto web; su SHA no sustituye al SHA del dataset científico JSON.

Para ampliar un paquete V0.1 con el AHC publicado:

```powershell
python -m web_app.backend.convert_ahc `
  --base C:\ruta\web-derived-v01 `
  --ahc C:\ruta\scientific-results\ahc-20261024T172400Z `
  --output C:\ruta\web-derived-v02-ahc
```

El destino debe ser nuevo. El conversor verifica manifiestos, schemas, hashes y recuentos antes de registrar `data_recalculated=false` y `scientific_artifacts_modified=false`.

## Desarrollo y pruebas

Backend, desde la raíz del repositorio:

```powershell
python -m pip install -r web_app/requirements-test.txt
$env:PYTHONPATH = (Get-Location).Path
$env:WEB_OFFICIAL_DATA_DIR = 'C:\ruta\web-derived-v01-20261024'
python -m pytest web_app/tests -q
```

Frontend:

```powershell
cd web_app/frontend
npm ci
npm run lint
npm run test -- --run
npm run build
```

Para desarrollo interactivo, iniciar Uvicorn en el puerto 8001 y Vite; el proxy de Vite dirige `/api` a ese backend.

## API v0.2

- `GET /api/health`
- `GET /api/stations`
- `GET /api/models`
- `GET /api/dataset/info`
- `GET /api/timeseries?station=IB04&from=2025-01-01&to=2025-12-31&models=EtPMon,PM_FAO56_V1`
- `GET /api/metrics?station=IB04&model=PM_FAO56_V1&group=global&cohort=common`
- `GET /api/scatter?station=IB04&model=HG_RS_V1&cohort=common`
- `GET /api/coverage?station=IB04`
- `GET /api/observations?station=IB04&limit=50&offset=0`
- `GET /api/ahc/models`
- `GET /api/ahc/timeseries?station=IB04&split=TEST`
- `GET /api/ahc/comparison?base_method=HG_RS_V1&split=TEST&station=IB04`
- `GET /api/ahc/coefficients?station=IB04&kind=AHC_VALIDATED`
- `GET /api/ahc/seasonal?station=IB04&split=TEST`
- `GET /api/ahc/coverage?station=IB04&split=ALL`

`ALL` devuelve las filas publicadas de TRAIN, VALIDATION y TEST por separado, sin crear una métrica conjunta. TRAIN es ajuste, VALIDATION selección y TEST `retrospective_locked`. `AHC_CLIMATOLOGY` se ofrece solo como coeficiente/descripción operacional, nunca como rendimiento TEST.

## Despliegue LAN

Antes de elegir puerto o rutas, comprobar `hostname`, `whoami`, `pwd`, `date`, `docker ps`, `docker compose ls`, `df -h` y `ss -lntp`. No reutilizar el checkout de SIAR Sync.

La instalación utiliza dos contenedores: backend de solo lectura y frontend Nginx. El único montaje de datos se declara `:ro`; ambos contenedores eliminan capabilities, aplican `no-new-privileges`, límites de memoria y healthchecks.

```bash
cp web_app/.env.example web_app/.env
# Editar WEB_DATA_DIR, WEB_BIND_IP y WEB_PORT con valores ya verificados.
docker compose --env-file web_app/.env -f web_app/compose.yaml config
docker compose --env-file web_app/.env -f web_app/compose.yaml up -d --build
docker compose --env-file web_app/.env -f web_app/compose.yaml ps
curl --fail http://DIRECCION_LAN:PUERTO/api/health
```

La aplicación se enlaza solo a la dirección LAN verificada. No abrir NAT, publicar el puerto en Internet ni integrar Forecast/AEMET/ML sin sus contratos científicos.

Para actualizar después de un merge: conservar el release derivado existente, actualizar el checkout propio, validar el nuevo paquete en un directorio de release nuevo, cambiar `WEB_DATA_DIR`, ejecutar `docker compose config`, reconstruir y verificar health/API/UI. No sobrescribir un release científico ni derivado existente.

## Documentos de decisión

- [Arquitectura y contrato](docs/architecture.md)
- [Revisión de la aplicación legacy](docs/legacy-review.md)
