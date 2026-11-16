"""Render an audit report from aggregate forensic results; no source/network I/O."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def render(results):
    results = Path(results)
    manifest = json.loads((results/"manifest.json").read_text(encoding="utf-8"))
    # This is an evidence-specific narrative, not an automatic unit classifier.
    # A different experiment must receive a fresh interpretation, even if its
    # data has the same columns. New directory names with identical artifacts
    # are supported; new snapshots cannot inherit these conclusions silently.
    expected_scores = "c7b3e98afac1f5749860bef6387ca19ffb3c76122fde5d4b036fc6d7d14da6c0"
    if manifest["output_sha256"]["scores.csv"] != expected_scores:
        raise ValueError("Narrative is specific to the reviewed units-offline-v2 scores; review new evidence")
    for name, digest in manifest["output_sha256"].items():
        if hashlib.sha256((results/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Output hash mismatch: {name}")
    with (results/"scores.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    def choose(corpus="canonical_snapshot", kernel="legacy_pm2", group="global", key=None):
        return [r for r in rows if r["corpus"] == corpus and r["kernel"] == kernel
                and r["group"] == group and (key is None or r["key"] == key)]
    def table(headers, data):
        return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)]
                         + ["| " + " | ".join(map(str, r)) + " |" for r in data]) + "\n"
    def number(row, key):
        return f"{float(row[key]):.6f}"
    def wins(kernel, group):
        grouped = defaultdict(list)
        for r in choose(kernel=kernel, group=group):
            grouped[r["key"]].append(r)
        if any(len({r["n"] for r in rs}) != 1 or len(rs) != 4 for rs in grouped.values()):
            raise ValueError("Unpaired hypothesis cohorts")
        winning = sum(min(rs, key=lambda r: float(r["rmse"]))["hypothesis"] == "Rad1_Wind1" for rs in grouped.values())
        return winning, len(grouped)
    sources = manifest["source_files"]
    canonical = [s for s in sources if s["corpus"] == "canonical_snapshot"]
    excluded = defaultdict(int)
    for source in canonical:
        for key, value in source["excluded"].items():
            excluded[key] += value
    text = """# UNITS-OFFLINE-FORENSIC — conclusión consolidada

**Interpretación respaldada: Radiacion en MJ m⁻² día⁻¹ y VelViento en m/s.**
No aplicar ×0.0864 ni /3.6 a estos snapshots. La conclusión combina el contrato
documentado en docs_1.1/docs_1.2, el uso directo legacy, la búsqueda de conversiones
y el contraste numérico amplio; no se selecciona una unidad por mínimo error.
**Cierre posterior de Dirección 00 (2026-10-19): RESOLVED_FOR_V1.**
Se acepta el uso directo de VelViento como u2 según convención SIAR de 2 m.
UNITS_PROVENANCE_PARTIAL y station_period_provenance no verificada de forma
exhaustiva se conservan como limitaciones no bloqueantes. Los diagnósticos STOP
preliminares de esta investigación quedan superados; solo evidencia positiva
contradictoria exige revisión. Este informe conserva el experimento, sin generar
Dataset v1 ni sus resultados finales. El cierre de integración se registra en
mission_status.md, separado de la fecha del experimento forense.

## Alcance y reproducibilidad

Solo lectura offline de los snapshots locales; ninguna consulta SIAR/API, ni
descarga, interpolación, imputación, calibración o cambio de datos. Se calculan
diagnósticos de hipótesis autorizados por 00, no Dataset Observational v1 ni sus
resultados finales. SIAR EtPMon es referencia calculada, no medición independiente.

Herramienta reproducible: `scientific_pipeline/forensics/units_offline.py`,
Python de biblioteca estándar. Desde la raíz del checkout:

```console
python scientific_pipeline/forensics/units_offline.py --repo . --output NUEVO_DIRECTORIO
python scientific_pipeline/forensics/report_units.py NUEVO_DIRECTORIO
python -m unittest discover -s scientific_pipeline/forensics -p "test_*.py" -v
```

El segundo comando escribe este informe en stdout. La herramienta exige un
directorio nuevo y rechaza destinos dentro de datos, legacy y adquisición.
El generador de informe está acotado por hash a los scores revisados de v2:
acepta otra ruta con esos mismos resultados y exige revisar la narrativa si
cambia la evidencia. No es un clasificador automático de unidades.
`results/units-offline-v2/manifest.json` registra SHA del checkout, estado dirty,
hash del script ejecutado, Python, catálogo, funciones PM2, cada CSV y cohortes.
El hash del script, y no solo el HEAD anterior a incorporarlo, identifica la
versión ejecutada. El manifest incluye hashes de las dos salidas numéricas.

**Bytes y finales de línea:** el checkout Windows usado tenía
`core.autocrlf=true`; los CSV y PM2 se leyeron con CRLF aunque Git almacena LF.
Los hashes del manifest describen esos bytes locales realmente leídos, no los
bytes originales del servidor ni una certificación POST. Un checkout LF puede
tener otro hash de archivo aun conservando los mismos valores. Los tests
verifican las fuentes reconstruyendo ambas representaciones en memoria, sin
reescribirlas; no admiten cambios de contenido. La herramienta registra en cada
ejecución su hash real, sin normalizarlo. `.gitattributes` solo dentro de la
carpeta forense preserva los bytes de las salidas y fija LF para sus scripts;
no cambia atributos, contenido o esquema de los CSV de origen.

Los tres corpus se mantienen separados, porque se solapan y no son muestras
independientes. No sumar sus n como si fueran observaciones distintas.

"""
    text += table(["Corpus", "Archivos", "Filas", "Pares elegibles"],
                  [(c, sum(s["corpus"] == c for s in sources),
                    sum(s["rows"] for s in sources if s["corpus"] == c),
                    sum(s["eligible"] for s in sources if s["corpus"] == c))
                   for c in ("canonical_snapshot", "legacy_calculated", "legacy_backup")])
    text += f"\nCanónicos: {len(canonical)} estaciones, {min(s['first'] for s in canonical)} a {max(s['last'] for s in canonical)}. "
    text += f"Se conservan {sum(s['observed_zero_included'] for s in canonical)} referencias EtPMon=0 en las métricas. "
    text += f"Exclusiones diagnósticas: {dict(excluded)}. "
    text += "Los periodos son los presentes en archivos; no se redefine el periodo oficial ni cobertura Quality.\n"
    text += f"El inventario físico de {sum(s['rows'] for s in canonical)} filas no sustituye los 75.494 días observados comunicados en PRE: este experimento no aplica el calendario oficial y no compara denominadores de cobertura.\n"
    text += """
La máscara común exige fecha parseable, inputs finitos, Tmin≤TempMedia≤Tmax,
RH extrema ordenada en [0,100], viento/radiación no negativos y temperaturas
en el dominio (-100,100) °C. No usa ±3σ, precipitación, suelo ni dirección.
No se eliminan ni reescriben filas: se cuentan exclusiones del experimento.
Duplicados o identidad de estación incoherente entre las filas elegibles
detienen la herramienta; las filas excluidas no pasan ese control. Esto no
sustituye los controles de integridad completos del Quality Audit.

## Hipótesis y fórmulas controladas

| Hipótesis | Rs de cálculo | u de cálculo |
|---|---|---|
| Rad1_Wind1 | Radiacion directa | VelViento directo |
| Rad1_Wind2 | Radiacion directa | VelViento / 3.6 |
| Rad2_Wind1 | Radiacion × 0.0864 | VelViento directo |
| Rad2_Wind2 | Radiacion × 0.0864 | VelViento / 3.6 |

0.0864=86400/10⁶ solo convierte W/m² como potencia **media durante 24 h** a
MJ m⁻² día⁻¹. No sería una conversión válida de una lectura instantánea sin
integración temporal. /3.6 convierte km/h a m/s. Son hipótesis explícitas del
experimento, no transformaciones aplicadas a los CSV.

`legacy_pm2` extrae por AST únicamente nueve funciones matemáticas de PM2.py,
sin importar ni ejecutar su I/O. Mantiene gamma variable, e° con 0.611,
273.15 K y su floor de nubosidad. `fao56_sensitivity` cambia a gamma=0.000665P,
e° con 0.6108, 273.16 K y cap superior Rs/Rso=1 sin floor inferior. Mantiene
TempMedia SIAR en ambos para aislar las hipótesis; no define con ello la futura
convención de temperatura del baseline. Se usa G=0 y RH extrema, sin calibración.
Dentro de cada kernel solo cambian los dos factores; todos los demás inputs,
metadatos, selección de días y supuesta altura de 2 m son idénticos.

Se usa el catálogo raíz para las métricas. Los placeholders IB03/IB04
(39.5°,100 m) se usan exclusivamente para comprobar los outputs históricos y
en una columna separada del ejemplo, nunca para mejorar el ajuste del contraste.

## Métricas globales por hipótesis

Bias=predicción−EtPMon; MAE y RMSE en mm/día, calculados sobre pares agrupados.
No son medias de métricas de estaciones. n es idéntico entre hipótesis.

"""
    text += table(["Corpus", "Kernel", "Hipótesis", "n", "bias", "MAE", "RMSE"],
                  [(r["corpus"], r["kernel"], r["hypothesis"], r["n"], number(r,"bias"), number(r,"mae"), number(r,"rmse"))
                   for r in rows if r["group"] == "global"])
    text += "\n## Estaciones canónicas: PM2, mismo catálogo y mismos pares\n\n"
    text += table(["Estación", "Hipótesis", "n", "bias", "MAE", "RMSE"],
                  [(r["key"], r["hypothesis"], r["n"], number(r,"bias"), number(r,"mae"), number(r,"rmse"))
                   for r in sorted(choose(group="station"), key=lambda r:(r["key"],r["hypothesis"]))])
    text += "\n## Robustez temporal y meteorológica\n\n"
    text += table(["Kernel", "Agrupación", "Grupos con menor RMSE Rad1_Wind1", "Grupos evaluados"],
                  [(k, g, *wins(k,g)) for k in ("legacy_pm2", "fao56_sensitivity")
                   for g in ("station", "year", "station_year", "season", "raw_radiation_bin", "raw_wind_bin", "temperature_bin", "humidity_bin")])
    exceptions = defaultdict(list)
    for r in choose(kernel="fao56_sensitivity", group="station_year"):
        exceptions[r["key"]].append(r)
    for key, cases in sorted(exceptions.items()):
        direct = next(r for r in cases if r["hypothesis"] == "Rad1_Wind1")
        best = min(cases, key=lambda r:float(r["rmse"]))
        if best["hypothesis"] == "Rad1_Wind1":
            continue
        comparison = {r["hypothesis"]: r for r in choose(group="station_year", key=key)}
        text += (f"\n**Excepción local conservada:** {key}, n={best['n']}. En fao56_sensitivity, "
                 f"{best['hypothesis']} obtiene RMSE {number(best,'rmse')} frente a {number(direct,'rmse')} "
                 f"con Rad1_Wind1. En PM2: {number(comparison[best['hypothesis']],'rmse')} frente a "
                 f"{number(comparison['Rad1_Wind1'],'rmse')}. Bajo la alternativa la radiación media "
                 f"sería {number(best,'rs_mean')} MJ/m²/día y Rs/Rso medio {number(best,'rs_rso_mean')}; "
                 f"Rnl es negativa en {best['negative_rnl']} días del kernel sin floor. "
                 "La onda larga negativa compensa parte de la disminución de Rs. "
                 "La sensibilidad del ranking al kernel y la cohorte corta impiden interpretar "
                 "este mínimo local como un cambio de unidad. No se oculta la discrepancia ni "
                 "se calibra contra ella. Es un ejemplo de por qué no elegir unidades solo por menor error.\n")
    text += """

Ese recuento es un contraste de consistencia entre hipótesis, no una prueba
estadística ni un criterio aislado de selección. Años parciales y periodos de
estaciones cerradas no se ponderan como años completos. El CSV contiene las
métricas completas por año, estación-año, estación climática y condiciones.
DJF/MAM/JJA/SON son estaciones meteorológicas del hemisferio norte, agrupadas
aquí sobre todos los años. Bins físicos previos: temperatura media ≤10,10–25,>25 °C;
RHmin≤40 o >40 %. Bins de escala bruta previos: radiación 0–5,5–20,>20 y viento
0–1,1–4,>4. Estos dos últimos conservan la escala fuente bajo TODAS las hipótesis,
para evitar cambiar la cohorte al convertir unidades; no se presentan como
clases físicas independientes de la hipótesis.

## Escala y diagnósticos físicos, canónicos

"""
    text += table(["Hipótesis", "Rs media MJ/m²/d", "u media m/s", "Rs/Rso medio", "ET0 media mm/d", "ET0 mín", "ET0 máx", "Rs/Rso bajo", "Rs>Rso", "ET0<0"],
                  [(r["hypothesis"], number(r,"rs_mean"),number(r,"wind_mean"),number(r,"rs_rso_mean"),number(r,"et0_mean"),
                    number(r,"et0_min"),number(r,"et0_max"),r["low_rs_rso"],r["rs_above_rso"],r["negative_et0"])
                   for r in choose()])
    text += """
Rs/Rso bajo significa <0.35/1.35: la expresión de nubosidad FAO sin floor cambia
de signo. Con Rad2 esta situación se generaliza; el kernel FAO puede producir
Rnl negativa y por ello amortiguar artificialmente parte de la caída de ET0.
El kernel legacy impide esa compensación mediante su floor; ambos rechazan
numéricamente la escala alternativa frente a EtPMon. No interpretar esta
comparación como validación del floor legacy para producción. Rs>Rso y ET0<0
son diagnósticos, no exclusiones adicionales ni correcciones silenciosas.

La reducción del viento /3.6 no crea por sí sola una velocidad imposible. Su
rechazo interpretativo combina la declaración explícita m/s en adquisición,
la ausencia de conversión y la pérdida sistemática de coherencia entre estaciones
y años. La escala de radiación, su relación con Rso y los cálculos diarios
documentados proporcionan evidencia adicional distinta del ranking de errores.

## Ejemplo IB04 y regresión histórica

"""
    examples = json.loads((results/"ib04_example.json").read_text(encoding="utf-8"))
    text += table(["Kernel", "Hipótesis", "EtPMon", "PM catálogo 39.562222°,8m", "PM placeholder 39.5°,100m"],
                  [(e["kernel"],e["hypothesis"],f"{e['observation']:.6f}",f"{e['catalog_metadata']['et0']:.6f}",
                    f"{e['historical_placeholder_metadata']['et0']:.6f}") for e in examples if e["corpus"] == "canonical_snapshot"])
    text += "\nIB04 2004-04-01: Radiacion=19.999, VelViento=1.908. El resultado 3.214339 corresponde al placeholder histórico, no al catálogo real. "
    text += f"La regresión de los seis outputs PM2 preservados reproduce {sum(s['eligible'] for s in sources if s['corpus']=='legacy_calculated')} filas "
    text += f"con discrepancia máxima {max(s['historical_reproduction_max_abs_error'] for s in sources if s['corpus']=='legacy_calculated'):.3g} mm/día. "
    text += "Eso verifica la extracción de funciones y explica el ejemplo; no certifica instrumentos.\n"
    text += """

## Interpretación, procedencia y altura: tres conclusiones distintas

1. **Unit interpretation: sólidamente respaldada para los snapshots auditados.**
   Radiacion es energía diaria MJ m⁻² día⁻¹ y VelViento está en m/s. Docs_1.1
   documenta la salida de adquisición, docs_1.2 la metodología y los programas
   preservados consumen esos campos directamente. La búsqueda exhaustiva no
   encuentra una conversión histórica observacional y el experimento amplio
   confirma coherencia de escala en múltiples estaciones, años y condiciones.
   No convertir de nuevo. No se extiende esta conclusión a futuras fuentes o
   lotes cuya escala cambie; seguir validando distribuciones y procedencia.
2. **Provenance certification: pendiente.** El manifest identifica los bytes
   efectivamente auditados y la evidencia de interpretación, pero no inventa
   historial de configuración del proveedor. El manual oficial previamente
   consultado admite configuraciones alternativas y no identifica la efectiva
   del lote. Esa limitación administrativa no obliga a mantener como desconocida
   una interpretación física fuertemente corroborada. Debe conservarse y
   completarse en el handoff POST; no certifica ni invalida Quality desde aquí.
3. **Wind sensor height: convención SIAR 2 m aceptada para v1.** Ninguna de las cuatro
   hipótesis varía altura; todas suponen 2 m al introducir u en la ecuación.
   Coincidir con EtPMon no demuestra altura real ni historia instrumental. Los
   catálogos locales no contienen ese campo y comentarios/plantillas no son
   fichas de sensor. La procedencia instrumental no exhaustiva se registra como
   limitación, sin investigación estación×año ni STOP global. Solo evidencia
   positiva de altura distinta exigiría una revisión versionada y resolución de
   la transformación afectada. No convertir altura por conjetura.

La documentación oficial ya consultada es compatible con MJ/m² y m/s en la
interfaz diaria, pero su manual API no garantiza por sí solo esa configuración
para el archivo histórico. Se combina como contexto, no como una certificación
que no proporciona. No se consultó API ni web adicional durante esta misión.

**Resultado forense, cerrado por 00:** RESOLVED_FOR_V1; no aplicar las dos
conversiones alternativas. Usar VelViento como u2 directo con
wind_height_assumption = 2 m / SIAR convention y station_period_provenance =
not exhaustively verified. UNITS_PROVENANCE_PARTIAL no bloquea la preparación.
La integración, CI y autorización de Dataset v1 se controlan por separado en
mission_status.md. No se solicita consulta SIAR adicional ni más investigación
histórica para discriminar estas unidades.

Evidencia y búsqueda detallada: [units_offline_evidence.md](units_offline_evidence.md).
Contrato actualizado: [contracts.md](contracts.md). Resultados agregados,
manifest y ejemplo: `../forensics/results/units-offline-v2/`.
"""
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = render(args.results)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
