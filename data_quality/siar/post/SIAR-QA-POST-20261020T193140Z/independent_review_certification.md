# Revisión independiente — POST superseding

**Resultado:** PASS  
**Audit ID:** SIAR-QA-POST-20261020T193140Z  
**Revisor:** revisión reproducible independiente del generador y sus salidas  
**Workspace:** /home/josep/quality-audit-post/SIAR-QA-POST-20261020T193140Z

## Conclusiones verificadas

1. El PRE original permanece inmutable: sus tres SHA coinciden con los valores registrados en el resumen original del POST.
2. Los cuatro artefactos del POST original permanecen byte a byte con los SHA aceptados.
3. IB07/2012-11-13 está en el raw una sola vez y coincide con FORENSIC-IB07-001/F2 después de canonicalizar el nombre de campo SIAR humedadMin a HumedadMin; la comparación canónica demuestra 23/23.
4. La fila ledger completa existe en el estado de reconciliación y en el ledger R2 persistido. En el POST superseding se cambian solo sus metadatos de auditoría (audit_id, repo_git_sha); los otros 16 campos coinciden con la evidencia.
5. Las 2.346 claves del POST corresponden exactamente a las 2.346 claves del estado de Recovery. Los 2.345 estados confirmados continúan siendo días sin observación y una fila documenta el dato recuperado.
6. El fix conserva los campos de Recovery desde el archivo persistido en todas las filas; el auditor valida el conjunto de ausencias, no vuelve a fechar su historia.
7. Los 12 SHA raw coinciden con el manifest; los 12 CSV tienen 23 columnas. El working tree del repo de datos permanece limpio en 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865.
8. IB09, CATALOG-TZ-001, cobertura y semántica de estados mantienen la decisión aceptada: Audit completeness COMPLETE; observational completeness INCOMPLETE (96.987410%); unresolved evidence 0; governance result COMPLETE_BUT_INCOMPLETE_DATASET.

## Resultado del verificador

verify_superseding_post.py pasó **2,438 assertions**. Las pruebas de gap_ledger.py pasaron **3/3**. Artefactos y conclusiones se detallan en final_consolidated_report.md; sus SHA-256 se conservan en superseding_checksums.sha256.

No hubo consultas SIAR ni cambios raw. No se declara Dataset Baleares v1 cerrado.

**Certificación final:** QUALITY-POST-RECOVERED-LEDGER-CORRECTION = PASS
