# VALIANTZAS_V1 — decisión científica de 2026-10-24

Decisión aprobada por Dirección 00 tras la revisión de fórmula y unidades:
adoptar la **composición de las ecuaciones 1b y 3 de Martí et al. (2024)**,
con conversión fija 0.408. No es una transcripción literal de su ecuación 4.
La elección precede al cálculo de errores contra EtPMon; no hay ajuste local.

## Fórmula y unidades

```text
Tcalc = (Tmax + Tmin) / 2                 [°C]
TD = Tmax - Tmin                        [°C]
RH = HumedadMedia                       [%; media diaria]
Ra = geometría solar FAO-56(fecha, latitud) [MJ m-2 día-1]
Rs_est = 0.338 Ra TD^0.3 (1.001-RH/100)^0.2 [MJ m-2 día-1]
ET0 = 0.0135 × 0.408 × Rs_est × (Tcalc+17.8) [mm día-1]
    = 0.001861704 Ra TD^0.3 (1.001-RH/100)^0.2 (Tcalc+17.8)
```

0.338 y los exponentes pertenecen a la estimación empírica de Rs; 0.0135
pertenece a la ecuación de Hargreaves con radiación solar; 17.8 es su offset
térmico. Estos coeficientes son empíricos y dependen de las unidades indicadas;
no son constantes universales adimensionales. **0.408 convierte energía diaria
en evaporación equivalente**, aproximadamente 1/λ con λ=2.45 MJ/kg y
1 kg/m²=1 mm. No transforma la columna SIAR Radiacion: este modelo no la usa.
No aplicar ×0.0864 a Radiacion ni /3.6 a VelViento.

Martí et al., p.5, ecuación 4, imprime el coeficiente 0.004563
(=0.0135×0.338), mientras define Ra en MJ m⁻² día⁻¹ y ET0 en mm/día.
La composición de 1b (incluye 1/λ) y 3 exige conservar la conversión energética.
0.004563 sería compatible con Ra ya expresada en mm equivalentes, pero no se
puede trasladar sin conversión a nuestra Ra energética. Registramos una
discrepancia interna de unidades en la expresión impresa, **sin atribuir a los
autores un erratum confirmado ni afirmar cómo calcularon sus resultados**.
No se ha verificado código autoral que resuelva la transcripción.

Usar exactamente 1/2.45 daría 0.0018624489795918368, un 0.040016 % mayor que
0.001861704; se fija 0.408 por coherencia con la convención FAO y HS_RA_V1.
Esta pequeña aproximación es explícita, no una calibración. Si se omite 0.408
con Ra energética, el resultado escala por 1/0.408≈2.45098.

## Contraste entre publicaciones y legado

| Fuente | Evidencia relevante | Decisión v1 |
|---|---|---|
| Martí et al. (2024), p.5, Ecs.1b y 3 | Rs estimada con RH media, offset **1.001** y exponente 0.2; ET0 con Rs/λ | Componer ambas ecuaciones con conversión explícita |
| Valiantzas (2018; online 2017) | Formulación de radiación estimada; offset 1.001 documentado | Conservar 1.001, no sustituir por 1 |
| Hargreaves y Allen (2003), p.54 | ET con Rs en evaporación equivalente y coeficiente 0.0135 para °C | Convertir primero la radiación energética |
| FAO-56, Ec.20 y Box 8 | MJ/m²/día → mm/día mediante ≈0.408 | Fijar el factor en parámetros/manifiestos |
| Moratiel et al. (2020), §2.3.2 Ec.2 | Ecuación con 0.0135 y 0.408 explícito cuando Ho está en MJ/m²/día | Comprobación independiente de la conversión; no sustituye la fórmula RH |
| `legacy/et0/et0alternativos.py` y `et0alternativo2.py` | 0.408×0.0135×0.338, factor **1−RH/100**, RH=HumedadMedia, T=TempMedia | Conservar legado; v1 cambia offset a 1.001 y usa Tcalc |

La lectura del legado se limita a trazabilidad: no se modifica ni se declara
que sus resultados históricos procedan de VALIANTZAS_V1. El comentario legacy
que trata 1.001 como errata no se adopta como evidencia contra la publicación.

## Dominio, elegibilidad y flags

Se necesitan Tmin, Tmax y HumedadMedia finitas, Tmin≤Tmax, ambas temperaturas
estrictamente superiores al cero absoluto (−273.15 °C), RH∈[0,100], fecha ISO
y −90<latitud<90 para Ra. La función acepta Ra finita ≥0. Este dominio físico
de temperatura es propio de VAL y no altera el dominio numérico PM/HS.
No se introduce un filtro de plausibilidad climático ni un sigma global.
Se conserva Quality PASS/WARN y observación OBSERVED/RECOVERED dentro de periodo.

No requiere viento, altura de anemómetro, altitud, Radiacion, RHmin/RHmax,
TempMedia o EtPMon. Este último es necesario solamente para evaluar: debe ser
finito y no negativo, incluyendo cero. No se infiere HumedadMedia desde extremos.
La ausencia de predicción es `null`/celda vacía con motivo; nunca cero imputado.
La elegibilidad nueva se calcula en los resultados; **no se reconstruye ni
modifica el Dataset Observational v1 existente**.

RH=100 conserva (0.001)^0.2 positivo; TD=0 o Ra=0 produce cero. No se hace
clipping de Rs estimada, humedad o ET0. Tcalc<−17.8 puede producir ET0 negativa:
se conserva con flag y se declara fuera del régimen agronómico habitual.
Flags adicionales: humedad media ausente/fuera de dominio, RH=100, TD=0 y
factor térmico negativo; se mantienen los diagnósticos de las otras variables.
Los fallos de cálculo dejan manifest FAILED y no un resultado oficial parcial.

## Alcance y limitaciones

Es una adaptación operacional explícita de ecuaciones publicadas, no una
reproducción certificada de la ecuación 4 ni de los resultados de Martí 2024.
La radiación es estimada empíricamente a partir de temperatura y humedad media;
no se valida su exactitud con el mero acuerdo de ET0 contra SIAR.
Las condiciones regionales, advección, agregación de RH y extrapolación al
frío/saturación pueden limitarla. ET0 presupone condiciones de referencia sin
limitación hídrica; no equivale a ET real de cualquier cultivo o suelo.
Sin calibración local, AHC ni parámetros aprendidos. EtPMon es referencia
calculada SIAR, no ground truth independiente.

Se retienen UNITS_PROVENANCE_PARTIAL, station_period_provenance no verificada
exhaustivamente, convención de viento SIAR 2 m, IB09 Quality FAIL excluido
científicamente pero conservado, y periodo corto IB101. Ninguna de estas
limitaciones se resuelve al añadir un método.

## Fuentes primarias

- [Martí et al. (2024), Agricultural Water Management, DOI 10.1016/j.agwat.2024.108903](https://doi.org/10.1016/j.agwat.2024.108903);
  [texto preservado por UPV](https://riunet.upv.es/server/api/core/bitstreams/e330a77f-f122-42a2-8aad-8448bdb01c2c/content).
- [Valiantzas (2018), DOI 10.1061/(ASCE)IR.1943-4774.0001275](https://doi.org/10.1061/(ASCE)IR.1943-4774.0001275).
- [Hargreaves y Allen (2003), DOI 10.1061/(ASCE)0733-9437(2003)129:1(53)](https://doi.org/10.1061/(ASCE)0733-9437(2003)129:1(53)).
- [FAO-56: conversión de radiación, Ec.20/Box 8](https://www.fao.org/4/X0490E/x0490e07.htm).
- [Moratiel et al. (2020), Natural Hazards and Earth System Sciences 20, 859–875](https://nhess.copernicus.org/articles/20/859/2020/nhess-20-859-2020.html).
