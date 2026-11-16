# IB09 Quality V2: excepción de DEVELOPMENT

Handoff `IB09-QUALITY-HANDOFF-V2-DEVELOPMENT-20261111` sustituye para DEVELOPMENT al snapshot PENDING de #67, que se conserva intacto y continúa rechazando admisión. La decisión expresa de Dirección 00 consta en `decision.md`; autoriza preparar una vista excepcional para IB09 con los límites inclusivos del catálogo, no lanzar campañas. El periodo oficial independiente sigue UNRESOLVED y Quality V1 sigue FAIL. Se excluye 2005-10-23 y las filas interiores son candidatas sujetas a controles por método.

El `source_repo_sha` fija el snapshot #66 y el `audit_source_repo_sha` identifica el commit auditado por POST. Los datos, catálogo, POST y ledger no se modifican. Las 4390 fechas comunes del inventario son preliminares y 05 deberá recalcular los resultados efectivos.

## Validación y responsabilidad de 05

Ejecutar `python data_quality/siar/ib09-handoff-v2-development/validate_handoff.py .` para validar evidencia y reconciliación. Para verificar condiciones de alcance, `--for-admission --scope DEVELOPMENT_ONLY --official-evidence-reviewed-no-contradiction`; esta última opción es una declaración explícita del llamador tras revisar fuentes disponibles, no una consulta automática ni una confirmación oficial. El validador falla si se indica contradicción, si no se declara revisión, si cambia un hash, si falla la reconciliación o si se pide otro alcance. Si aparece evidencia oficial que contradiga el catálogo, se detiene la admisión y se informa a Dirección 00.

El validador verifica el hash de la decisión, el handoff PENDING y su validador, y reutiliza la reconciliación de #67. No comprueba que un dataset derivado contenga los valores de las filas originales ni ejecuta métodos científicos. 05 debe derivar o cotejar cada fila contra los bytes raw verificados, preservar FAIL y flags V1, registrar en linaje el ID y SHA256 del handoff, y aplicar filtros por estación, método, periodo y experimento. El resultado `valid` valida este handoff, no constituye validación científica de filas ni permiso para iniciar campañas.

`decision.md` registra la instrucción directa de Dirección recibida en el hilo 05; la hora expresada es el inicio del turno registrado por la aplicación. El snapshot oficial sigue sin recuperarse. El esquema es cerrado: cualquier cambio de evidencia o autorización exige una nueva revisión versionada.
