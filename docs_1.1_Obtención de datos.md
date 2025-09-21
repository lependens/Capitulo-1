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

 ### siarIBconsulta2.py

 Con este programa tenemos como inputs:

-ESTACIÓN: "IB01" (POR EJEMPLO)
-FECHA INICIO: ""dd""mm""aaaa""
-FECHA FIN: ""dd""mm""aaaa""








