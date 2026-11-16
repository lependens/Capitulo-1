# HG_RS_V1 — Hargreaves con radiación solar medida

Decisión de 2026-10-24: incorporar un cuarto baseline clásico independiente,
sin calibración, conservando Dataset Observational v1 y las ejecuciones previas.

## Fórmula versionada

```text
method_id = HG_RS_V1
Tcalc = (TempMax + TempMin)/2                    [°C]
Rs = Radiacion SIAR                             [MJ m-2 día-1]
ET0 = 0.0135 × 0.408 × Rs × (Tcalc + 17.8)        [mm/día]
    = 0.005508 × Rs × (Tcalc + 17.8)
```

La ecuación de Hargreaves con radiación solar utiliza el coeficiente 0.0135
para temperatura en °C y Rs expresada en evaporación equivalente. Con nuestra
Rs energética hay que conservar 0.408≈1/2.45: 1 kg/m² evaporado equivale a 1 mm.
No convertir Radiacion desde W/m² ni aplicar 0.408 dos veces.
El coeficiente empírico no es universalmente adimensional: depende de las
unidades y de la definición de temperatura/radiación fijadas aquí.

## Distinción entre los cuatro métodos

| Método | Radiación / información utilizada | Expresión principal |
|---|---|---|
| PM_FAO56_V1 | Rs medida, humedad, viento, temperaturas y geometría | FAO-56 PM diario |
| HS_RA_V1 | Ra geométrica y rango térmico como estimador de radiación | 0.0023×0.408×Ra×(Tcalc+17.8)×√TD |
| VALIANTZAS_V1 | Ra, rango térmico y RH media estiman Rs | Martí Ecs.1b+3 con 0.408 |
| HG_RS_V1 | **Rs medida** y Tcalc | 0.0135×0.408×Rs×(Tcalc+17.8) |

No se sustituye Ra por Rs dentro de la fórmula HS_RA_V1: desaparece el
factor √TD y cambia el coeficiente. El valor 0.0023 de HS es la constante
convencional redondeada; no se reestima a partir de HG-Rs ni se modifica.
La geometría solar no es un input de HG-Rs. Los diagnósticos generales QC
pueden calcular Rs/Rso, pero no transforman Rs medida ni condicionan HG-Rs.

## Entradas, dominio y elegibilidad

- Predicción en pipeline: TempMin, TempMax y Radiacion finitas; Tmin≤Tmax;
  temperaturas >−273.15 °C; Rs≥0. Tcalc se deriva de extremos y TempMedia
  SIAR se conserva. La función pura recibe únicamente Tcalc y Rs.
- Se mantienen estados OBSERVED/RECOVERED dentro de periodo, Quality PASS/WARN
  y fecha válida del registro. Latitud válida sigue siendo parte del contrato
  general Dataset v1, pero HG-Rs no la necesita como entrada física ni para
  su elegibilidad específica. No necesita altitud, viento, altura del viento,
  humedad, Ra o EtPMon para predecir.
- EtPMon finita/no negativa se necesita solo para evaluar. Cero incluido.
- Rs ausente, negativa o no finita deja `null` con motivo; no se estima desde
  Ra, TD, humedad ni otro modelo. No se imputan temperaturas ni se usa
  TempMedia para rescatar extremos ausentes.
- Rs=0 produce ET0=0; **Tmin=Tmax no implica ET0=0** si Rs>0. No hay potencia
  de TD. Tcalc<−17.8 puede generar ET0 negativa: se conserva y marca mediante
  `negative_prediction:HG_RS_V1`. No clipping, sigma global ni calibración.
- Una altura de viento incompatible puede detener PM; no bloquea HG-Rs.
  Los flags generales se preservan aunque una variable no sea entrada HG-Rs.
- Fallos aritméticos estructurales dejan manifest FAILED. La elegibilidad
  nueva se calcula en resultados: no se reconstruye el dataset existente.

## Trazabilidad histórica y fuentes

`legacy/et0/et0alternativo2.py`, líneas 28–32, ya implementaba
`ET0_harg_rs = 0.0135 * 0.408 * Radiacion * (TempMedia + 17.8)`.
`calculate_ahc2.py` y `calculate_errors2.py` distinguían Hargreaves Ra/Rs,
incluidas versiones ajustadas. Ese cálculo pertenece al historial: no se
copian sus scripts interactivos, filtros, AHC, métricas o rutas hardcoded.
La diferencia térmica explícita v1 es TempMedia→Tcalc de extremos, coherente
con PM/HS/VAL. Sus resultados legacy no se reetiquetan como HG_RS_V1.

- [Hargreaves y Allen (2003), DOI 10.1061/(ASCE)0733-9437(2003)129:1(53)](https://doi.org/10.1061/(ASCE)0733-9437(2003)129:1(53)): desarrollo histórico y formulación con radiación solar.
- [Parametric expressions for the adjusted Hargreaves coefficient in Eastern Spain, manuscrito UPV, Ecs.4–5](https://riunet.upv.es/bitstreams/21e22951-21aa-41bc-bcea-4b05b2d642d2/download): 0.0135×Rs×(T+17.8), Rs en mm equivalentes, distinta de estimar Rs a partir de Ra y TD.
- [FAO-56, Ec.20 y Box 8](https://www.fao.org/4/X0490E/x0490e07.htm): conversión MJ/m²/día→mm/día mediante 0.408.
- [USACE HEC-HMS, Hargreaves Method](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/evaporation-and-transpiration/hargreaves-method): distingue la ecuación con Rs de su combinación con un estimador Ra/TD.

## Interfaz y compatibilidad

```text
calculate-hargreaves-rs --dataset DATASET --output NUEVO_RUN_HGRS
evaluate-four --dataset DATASET --predictions TRES_OFICIALES/predictions.json
              --predictions NUEVO_RUN_HGRS/predictions.json --output NUEVA_COMPARACION
run-four-baselines --dataset DATASET --output NUEVO_RUN_CUATRO
```

Invocar con `python -m siar_science` o la CLI instalada. La evaluación importa
predicciones exitosas, sin solapamientos, verifica parámetros y ambas
identidades audit/canonical, y conserva productores y hashes upstream.
`run-classical-baselines`/`evaluate-classical` siguen significando tres métodos;
`run-baselines`/`evaluate` mantienen sus convenciones PM/HS anteriores.

La comparación nueva informa cohortes propias, seis pares, cuatro triples y
**PM_HS_VAL_HGRS** como comparación principal de cuatro métodos. Todas con
n y hashes de claves estación-fecha; desgloses global/estación/año/mes/temporada,
estación-año y temporada-año. No se redefine `paired` ni PM_HS_VAL.
Se añade ET0_HG_RS_V1 y eligible/evaluated/eligibility_reasons_HGRS al CSV
de cuatro métodos, conservando las 23 columnas originales y todas las filas.
Los comandos de tres métodos conservan su esquema CSV previo.

## Límites

Se depende de la calidad de Rs medida y temperaturas. Una menor dependencia
de estimar radiación no garantiza mayor exactitud ET0: la fórmula sigue siendo
empírica y carece de términos explícitos de viento/humedad/advección.
Es ET de referencia bajo condiciones sin limitación hídrica, no ET real de
cualquier cultivo. EtPMon es referencia calculada SIAR y comparte Rs con PM
y HG-Rs: la comparación no prueba exactitud frente a mediciones independientes.

Se mantienen UNITS_PROVENANCE_PARTIAL, station_period_provenance no verificada
exhaustivamente, viento 2 m convencional, IB09 Quality FAIL excluido de cálculo
pero preservado y muestra corta IB101. Sin AHC, ajuste local, ML o forecast.
