# Corrección del ledger de Quality POST

## Dictamen

El POST original SIAR-QA-POST-20261018T233901Z se conserva byte a byte y queda superseded por SIAR-QA-POST-20261020T193140Z para la lectura del ledger. No se ejecutaron Recovery, consultas SIAR ni una nueva auditoría raw.

## Causa raíz

El generador inicial usó como plantilla de salida la lista de fechas ausentes emitida por el auditor. Esa lista excluye, por definición, fechas que ya están presentes en el CSV. IB07 / 2012-11-13 ya estaba en el raw validado, por lo que la fila RECOVERED del ledger persistido no se copió al informe POST. El generador además estampaba first_detected_at_utc con la hora del auditor, en vez de preservar el valor histórico del ledger de trabajo.

El arreglo merge_gap_ledger valida la correspondencia entre fechas ausentes y el ledger persistido, emite cada fila persistida, preserva todos los campos históricos y actualiza únicamente audit_id y repo_git_sha para el nuevo POST. Rechaza duplicados, estados contradictorios y ausencias sin estado justificante. Se añadieron pruebas de regresión y se conectaron a CI.

## POST superseding

- ID: SIAR-QA-POST-20261020T193140Z
- Supersedes: SIAR-QA-POST-20261018T233901Z
- audit_source_repo_sha: 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865
- expected: 77,840; observed: 75,495; coverage: 96.987410 %
- RECOVERED: 1; CONFIRMED_NO_DATA: 2,345; PENDING_RETRY: 0
- Resultado: FAIL 1 (IB09 fuera del periodo), WARN 11, INCOMPLETE 0
- Revisión independiente: PASS, 2,438 assertions

La fila IB07 / 2012-11-13 coincide 23/23 con FORENSIC-IB07-001/F2, comparación canónica independiente; registro presente una vez. La procedencia de la escritura original sigue sin resolverse; la validez del contenido está demostrada. IB09 / 2005-10-23 se mantiene intacta y fuera del periodo oficial. CATALOG-TZ-001 sigue OPEN / NON_BLOCKING.

Los tres CSV POST se copiaron desde el workspace corregido y sus SHA-256 se verificaron tras la copia. El POST original permanece en su ruta y no se sobrescribió. Sin cambios raw.
## Hashes POST corregidos

- siar_station_audit.csv: e444cebf4165939c24cb0e9c132b241363bb60b0808e4403e1cf2ac33bd6be7d
- siar_missing_days.csv: 55a78e6affff09e2a1e8cb28e0465b2073ad79a1e573a1efe272a52462ba3969
- siar_dataset_manifest.csv: b37c6ef2cf47f4ac7ac7308d0414b39429c5a35c352ec09581f9f4751468337b
