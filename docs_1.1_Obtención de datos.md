# 1. Obtención de datos históricos y actuales de las estaciones meteorológicas de Baleares

**Objetivo:** Conseguir todos los datos históricos y actualizados hasta 2024 de las estaciones meteorológicas de las Islas Baleares de la pagina web del SIAR de manera automatizada, para almacenarlos en una **base de datos local** en formato CSV.

**A CONTINUACIÓN SE INDICAN LOS PASOS QUE SE VAN DANDO**


## 09/09/2025: Con siarID he podido conseguir una lista de todas las estaciones de España, con los siguientes datos:

1. Nombre estación
2. Código estación
3. Termino 
4. Longitud 
5. Latitud 
6. Altitud
5. Fecha de alta
6. Fecha de baja

El siguiente objetivo: Filtrar estaciones de Islas Baleares

## 20/09/2025: Hacer un programa en python para obtener solamente las estaciones de baleares.

Con siarIDIB hacemos el filtro y lo exporta en fomrato csv ya filtrado.

Vamos a intentar ir consigiendo datos, sabiendo ya que no podemos hacer consultas masivas ya que excederemos el límite. 

Por tanto, vamos a intentar hacer un programa que recoja los datos historicos de las estaciones de las Islas BAleares.

### siarIBconsulta.py

Tras ajustar el codigo con ayuda de GROK, he detectado que no puedo hacer la consulta masiva de todas a la vez. 

Por tanto he tomado los siguientes criterios:

1. Empezar con estación IB01
2. Recoger datos mensuales para no saturar la consulta
3. Compilarlos posterormente en un mismo archivo

Consideraciones:

- IB01 Se inserta manuelamente (Posteriormente cambiaremos a IB02, IB03,...)
- La fecha inicio la extrae de la columna pertinente en el archivo estaciones_baleares
- La fecha fin es 2024-12-31 , a no ser que tenga menos registros, previamente indicado en el archivo estaciones_baleares (Debo comprobar qye es así, o si por defecto, si no encuentra en fecha, simplemente da error y el archivo incluye hasta el maximo de dias recogidos)

Resultado: Se ha podido extraer los datos , hsata que se superó el limite diario, y se quedó en 2023. Se ha ceado un archivo IB01_datos_completos.csv. Los datos de momento no se han analizado. Simplemente se pueden observar fallos al abrir en CSV desde Excel , con valores com comas corridas y demás. Se evaluará posteriormente los datos para ajustarlos y ponerlo todo en un formato coherente y filtrarlos.

#### Como conclusión de siarIBconsulta:
 Vamos a hacer un programa que coga de la misma manera los datos, pero con las siguietnes consideraciones:

 Inputs:
 -ESTACIÓN: "IB01" (POR EJEMPLO)
 -FECHA INICIO: ""dd""mm""aaaa""
 -FECHA FIN: ""dd""mm""aaaa""

 Cuando vaya a guardar los datos en la carpeta "datos_siar_baleares" que los guarde como IB01_datos_completos.csv con los isguientes condicionantes:
 -Si no existe dicho archivo (dependiende de si existe ya datos de esa estación), crea archivo


 -Si existe dichoa rchivo, lo actualiza si las fechas que acabamos de descargar no eran existentes en ese archivo. Por tanto comprueba el archivo que datos tiene actualmente


 Justificación: Podremos así ir obtieniendo cada dia los datos poco a poco e ir completandolos o actualizandolos

 ### siarIBconsulta2_corregido.py

 Con este programa tenemos como inputs:

-ESTACIÓN: "IB01" (POR EJEMPLO)
-FECHA INICIO: ""dd""mm""aaaa""
-FECHA FIN: ""dd""mm""aaaa""

 -Si existe dicho archivo, lo actualiza si las fechas que acabamos de descargar no eran existentes en ese archivo. Por tanto comprueba el archivo que datos tiene actualmente.

 El formato de salido es el siguiente:
 IBXX_datos_completos.csv

| Variable              | Unidad             |
|-----------------------|--------------------|
| Fecha                 | YYYY-MM-DD         |
| TempMedia             | °C (Grados Celsius)|
| TempMax               | °C (Grados Celsius)|
| HorMinTempMax         | Minutos            |
| TempMin               | °C (Grados Celsius)|
| HorMinTempMin         | Minutos            |
| HumedadMedia          | % (Porcentaje)     |
| HumedadMax            | % (Porcentaje)     |
| HorMinHumMax          | Minutos            |
| HumedadMin            | % (Porcentaje)     |
| HorMinHumMin          | Minutos            |
| VelViento             | m/s (Metros por segundo) |
| VelVientoMax          | m/s (Metros por segundo) |
| HorMinVelMax          | Minutos            |
| DirViento             | ° (Grados, 0-360)  |
| DirVientoVelMax       | ° (Grados, 0-360)  |
| Precipitacion         | mm (Milímetros)    |
| PePMon                | mm (Milímetros)    |
| Radiacion             | MJ/m² (Megajulios por metro cuadrado) |
| TempSuelo1            | °C (Grados Celsius)|
| TempSuelo2            | °C (Grados Celsius)|
| EtPMon                | mm (Milímetros)    |
| Estacion              | Código de estación |





---

# ACTUALIZACIÓN 2026 — SIAR Sync

Durante 2026 el sistema de obtención de datos ha evolucionado desde los primeros scripts de consulta desarrollados durante 2025 hacia un sistema automatizado de sincronización de datos.

El objetivo deja de ser simplemente descargar datos y pasa a ser:

> **Mantener una copia local fiable, verificable y actualizable de los datos meteorológicos diarios de las estaciones SIAR de las Islas Baleares.**

Esta nueva etapa se desarrolla mediante un sistema independiente denominado **SIAR Sync**.

---

## 1. Acceso a la API oficial de SIAR

Tras las primeras pruebas realizadas mediante scripts independientes, se decidió trabajar directamente con la API oficial de SIAR.

La API permite consultar:

* información de estaciones;
* datos meteorológicos;
* información de acceso;
* diferentes parámetros de consulta.

La dirección utilizada actualmente es:

```text
https://servicio.mapa.gob.es/siarapi
```

La consulta de datos diarios se realiza mediante el endpoint correspondiente a:

```text
/API/V1/Datos/{tipoDatos}/{ambito}
```

utilizando parámetros de fecha para limitar cada consulta.

La autenticación se realiza mediante el sistema de tokens proporcionado por SIAR.

---

## 2. Problema de las consultas masivas

Una de las principales dificultades detectadas durante las primeras versiones fue que no era viable realizar una única consulta masiva para descargar todos los datos históricos de una estación.

La API dispone de límites de acceso y de registros.

Por este motivo, el sistema actual no intenta descargar grandes períodos de una sola vez.

En su lugar, divide las solicitudes en pequeños bloques temporales.

Actualmente se utiliza un máximo de:

```text
7 días por bloque
```

Esto permite controlar mejor el número de registros descargados y limitar las consecuencias de un posible error.

---

## 3. Consulta de los límites de la API

Una mejora importante respecto a los primeros scripts consiste en no asumir que los límites de acceso son constantes.

SIAR proporciona información sobre los límites mediante sus servicios de información.

Entre los parámetros disponibles se encuentran:

* accesos realizados;
* máximo de accesos por minuto;
* accesos diarios;
* máximo de accesos diarios;
* registros acumulados por minuto;
* máximo de registros por minuto;
* registros acumulados diarios;
* máximo de registros diarios.

El sistema utiliza esta información para conocer el estado de acceso y evitar diseñar el proceso basándose en valores históricos que podrían cambiar.

---

## 4. Cobertura real de los archivos locales

Una de las mejoras fundamentales de SIAR Sync es que el sistema no considera que un archivo esté actualizado simplemente porque exista.

Antes de descargar información se analiza el contenido real del CSV.

Para cada estación se comprueba:

* primera fecha disponible;
* última fecha disponible;
* número de registros;
* número de fechas únicas;
* duplicados;
* huecos internos;
* fechas pendientes.

Esto permite conocer la cobertura real de cada estación.

---

## 5. Detección de huecos internos

Una situación especialmente importante aparece cuando un archivo contiene datos antiguos y recientes, pero existe un período intermedio sin información.

Por ejemplo:

```text
2004 ───────── 2010
                  │
                  │  ← hueco
                  │
2025 ─────────────
```

Un sistema basado únicamente en la primera y última fecha podría considerar erróneamente que el archivo está completo.

SIAR Sync analiza las fechas existentes para detectar estos huecos.

De esta forma puede solicitar únicamente los períodos que faltan.

---

## 6. Descarga únicamente de fechas pendientes

El sistema genera una lista de fechas que todavía no están correctamente cubiertas.

Posteriormente agrupa esas fechas en bloques de consulta.

Antes de realizar cada descarga se vuelve a leer el archivo local.

Esto es importante porque el estado del archivo puede haber cambiado entre dos consultas.

La lógica general es:

```text
Leer CSV
   ↓
Analizar fechas
   ↓
Detectar pendientes
   ↓
Crear bloque
   ↓
Volver a comprobar CSV
   ↓
¿El bloque sigue pendiente?
   │
   ├── No → Saltar
   │
   └── Sí → Consultar SIAR
```

De esta forma se reduce la posibilidad de descargar dos veces la misma información.

---

## 7. Validación de la respuesta de SIAR

La respuesta recibida de la API no se incorpora directamente al archivo.

Primero se realizan comprobaciones básicas:

* respuesta válida;
* estructura esperada;
* existencia de registros;
* fechas dentro del intervalo solicitado;
* datos correspondientes a la estación solicitada.

Si una consulta devuelve cero registros, el sistema no inventa información ni genera filas artificiales.

La ausencia de datos se conserva como ausencia de datos.

---

## 8. Estructura canónica de los CSV

Durante el desarrollo se estableció una estructura fija para los archivos históricos.

La estructura canónica actual está formada por 23 columnas:

```text
Fecha
TempMedia
TempMax
HorMinTempMax
TempMin
HorMinTempMin
HumedadMedia
HumedadMax
HorMinHumMax
HumedadMin
HorMinHumMin
VelViento
DirViento
VelVientoMax
HorMinVelMax
DirVientoVelMax
Radiacion
Precipitacion
TempSuelo1
TempSuelo2
EtPMon
PePMon
Estacion
```

La API puede proporcionar determinados campos adicionales, como códigos asociados a la temperatura del suelo, pero estos no forman parte del formato canónico de los CSV.

También se normalizan diferencias de nomenclatura cuando es necesario, por ejemplo:

```text
humedadMin → HumedadMin
```

---

## 9. Deduplicación y orden cronológico

Una vez obtenidos nuevos registros, estos se combinan con los datos existentes.

Antes de guardar el resultado se:

1. combinan los datos;
2. identifican duplicados por fecha;
3. conservan un único registro por fecha;
4. ordenan cronológicamente;
5. comprueba la estructura final.

El objetivo es que cada estación disponga de una serie diaria coherente y sin duplicaciones.

---

## 10. Escritura atómica de los archivos

Otra mejora importante respecto a los primeros scripts es la forma de guardar los datos.

El archivo original no se sobrescribe directamente durante todo el proceso.

Primero se genera un archivo temporal con el resultado completo.

Después se valida.

Solo cuando la validación es correcta se sustituye el archivo original mediante una operación atómica.

Conceptualmente:

```text
CSV original
     │
     ├── leer
     │
     ├── actualizar
     │
     ├── validar
     │
     └── guardar temporal
              │
              ↓
        validación correcta
              │
              ↓
        sustitución atómica
```

Esto reduce el riesgo de dejar un CSV parcialmente escrito si se produce un error durante el proceso.

---

## 11. Actualización de datos modificados

El sistema no solamente busca fechas que no existen.

También contempla la posibilidad de que SIAR modifique o complete registros que ya estaban descargados.

Para ello se puede utilizar el parámetro de modificación disponible en la API:

```text
FechaUltModificacion
```

Esto permite plantear una actualización incremental en la que se revisen datos que hayan sido modificados posteriormente.

---

## 12. Diferenciación entre ausencia de datos y error de descarga

Una de las reglas importantes del sistema es no interpretar automáticamente una ausencia de registros como un fallo.

Puede ocurrir que una fecha:

* no tenga datos disponibles;
* todavía no haya sido incorporada por SIAR;
* haya sido solicitada incorrectamente;
* provoque un error temporal de conexión.

Estas situaciones deben diferenciarse.

Por tanto, el sistema evita rellenar artificialmente las fechas sin datos.

---

## 13. Autenticación y control de errores

La autenticación con SIAR se realiza mediante token.

Durante el desarrollo se detectó la necesidad de diferenciar claramente:

* token inválido o inexistente;
* límite de consultas;
* error temporal de red;
* error de servidor;
* ausencia real de datos.

Esto es importante porque cada problema requiere una estrategia diferente.

Por ejemplo, un error de autenticación no debe tratarse como si fuese un problema de límite de consultas.

---

## 14. Integración con Git

Una vez completada una actualización local, el sistema puede utilizar Git para registrar los cambios.

El flujo es:

```text
SIAR
 ↓
SIAR Sync
 ↓
CSV actualizado
 ↓
Validación
 ↓
Git status
 ↓
Commit
 ↓
Push
```

Esto permite conservar un historial de los cambios de los datos.

Además, los datos locales no se consideran perdidos si GitHub no está disponible temporalmente.

---

## 15. Recuperación ante errores de GitHub

Durante las pruebas reales apareció un problema de conexión TLS/GnuTLS al realizar determinados `git push`.

La investigación permitió diferenciar este problema de los errores de autenticación de SIAR.

Para evitar que un fallo temporal de red impida completar el proceso local, se incorporó una estrategia de reintentos para errores transitorios de Git.

Los reintentos utilizan esperas progresivas:

```text
0 s
5 s
15 s
30 s
```

hasta un máximo de cuatro intentos.

El mecanismo se aplica únicamente a errores considerados potencialmente transitorios.

Los errores que no corresponden a este tipo de fallo no se repiten indefinidamente.

---

## 16. Ejecución mediante Docker

El sistema se ejecuta actualmente dentro de un contenedor Docker.

La arquitectura permite mantener:

* aplicación;
* worker;
* configuración;
* dependencias;
* entorno de ejecución;

separados del sistema operativo principal.

El contenedor se denomina actualmente:

```text
siar_sync
```

y la aplicación web de control utiliza el puerto:

```text
8000
```

La interfaz permite consultar el estado de las estaciones y del proceso de sincronización.

---

## 17. Arquitectura actual

La arquitectura de la etapa de adquisición puede resumirse de la siguiente forma:

```text
                 API SIAR
                    │
                    ↓
             Autenticación
                    │
                    ↓
             SIAR Sync Worker
                    │
          ┌─────────┴─────────┐
          │                   │
     Cobertura             Pendientes
          │                   │
          └─────────┬─────────┘
                    ↓
             Bloques ≤ 7 días
                    │
                    ↓
              Consulta SIAR
                    │
                    ↓
               Validación
                    │
                    ↓
              Deduplicación
                    │
                    ↓
             CSV actualizado
                    │
              ┌─────┴─────┐
              ↓           ↓
           Git local    Aplicación
              │
              ↓
           GitHub
```

---

## 18. Estado alcanzado

El sistema ya ha permitido recuperar y consolidar una cantidad importante de información histórica de las estaciones de Baleares.

Como ejemplo de las pruebas realizadas durante esta etapa, la estación IB04 ha alcanzado cobertura hasta finales de 2025, con miles de registros diarios, sin duplicados y manteniendo la estructura canónica de 23 columnas.

También se han identificado fechas pendientes para las que SIAR no devuelve necesariamente datos.

Esto es importante porque el objetivo no es conseguir artificialmente una serie sin huecos, sino conocer con precisión qué información existe realmente en SIAR.

---

## 19. Estado del proyecto de adquisición

La situación actual puede resumirse como:

```text
Descarga manual
      ↓
Scripts experimentales
      ↓
Consultas mensuales
      ↓
Actualización incremental
      ↓
API oficial
      ↓
Detección de huecos
      ↓
Validación
      ↓
SIAR Sync
      ↓
Docker + Git
```

El sistema de adquisición histórica se considera actualmente una infraestructura estable sobre la que pueden desarrollarse las siguientes fases del proyecto.

---

## 20. Próxima etapa: recopilación de predicciones

La recopilación de predicciones se tratará como un proyecto paralelo a SIAR Sync.

La arquitectura prevista será:

```text
                 SIAR
                  │
        ┌─────────┴─────────┐
        │                   │
   Datos históricos     Predicciones
        │                   │
    SIAR Sync         Forecast Collector
        │                   │
        └─────────┬─────────┘
                  ↓
             Dataset final
```

No se modificará SIAR Sync para incorporar esta nueva función mientras no sea necesario.

El objetivo del nuevo sistema será investigar:

* si las predicciones de ET₀ están disponibles mediante API;
* qué variables meteorológicas proporciona SIAR;
* qué horizonte de predicción utiliza;
* cuándo se genera cada predicción;
* cómo identificar cada ejecución;
* cómo conservar las predicciones históricas;
* cómo compararlas posteriormente con las observaciones reales.

Las predicciones deberán almacenarse de forma que una nueva actualización no sobrescriba las predicciones emitidas anteriormente.

Esto permitirá posteriormente estudiar el error de predicción para diferentes horizontes temporales.

---

## 21. Próximo objetivo de los datos

Una vez consolidada la adquisición histórica, el siguiente objetivo será transformar los archivos CSV en un dataset preparado para análisis y Machine Learning.

La arquitectura prevista será:

```text
CSV SIAR
   ↓
Validación
   ↓
Normalización
   ↓
Parquet
   ↓
DuckDB / Python
   ↓
Modelos ET₀
```

El CSV seguirá siendo una fuente sencilla y auditable, mientras que Parquet y DuckDB podrán utilizarse para análisis más grandes y eficientes.

---

> **Conclusión de la actualización 2026:**
> La obtención de datos ha pasado de ser un conjunto de scripts experimentales a convertirse en un sistema automatizado de sincronización. El objetivo actual no es únicamente descargar datos, sino garantizar su cobertura, integridad, trazabilidad y capacidad de actualización, creando una base sólida para las siguientes fases de análisis de ET₀, Machine Learning y predicción.




