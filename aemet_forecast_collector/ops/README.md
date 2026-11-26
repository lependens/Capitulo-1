# AEMET backup OAuth persistence

The backup service runs as `aemet-forecast` and uses its own writable rclone
configuration at `/var/lib/aemet-forecast/rclone/rclone.conf`. OAuth refreshes
must survive process exit. The directory is owned by `aemet-forecast`, mode
0700; the configuration is mode 0600. Keep credentials and crypt recovery keys
out of Git and logs. The Google provider and the primary crypt password are
shared with SIAR by owner decision; AEMET uses its own second crypt password and
logical destination. Directory separation does not restrict the OAuth account.

The service retains `ProtectSystem=strict`, `NoNewPrivileges=yes`, and its
existing restrictions. Its only added writable path is the dedicated rclone
directory. Capture still receives its API key via systemd `LoadCredential`.

## Deployment and validation

Deploy only with explicit operational authorization. First preserve the active
`ops.py`, backup unit and AEMET rclone configuration in a root-only directory;
record hashes, ownership and modes. Verify that no service is using the files.
Provision the dedicated 0700 directory and a 0600 copy of the AEMET config,
owned by the AEMET account. Never copy or change the SIAR configuration.
Install the reviewed `ops.py` and backup unit, verify the unit and reload
systemd without enabling the capture timer. Confirm the configured sandbox,
permissions and gate, and compare crypt settings without disclosing secrets.

Verify OAuth refresh by observing an actual expired access token being renewed
and persisted in the AEMET configuration, without revocation or logging token
values. An independent subsequent process under the service identity and
equivalent systemd restrictions must access the crypt remote. For end-to-end
validation copy a uniquely named synthetic file, verify the remote decrypted
bytes, restore to an isolated location and compare SHA-256. Do not use `sync`
or modify published raw runs. A synthetic test is not a real capture test.

## Rollback

Keep AEMET capture disabled. Restore the saved operational script and unit with
their original ownership and modes, verify them and reload systemd. Preserve the
persistent configuration and protected original until credential state is
reviewed; do not blindly roll back OAuth tokens, which may have been invalidated
or rotated. Do not delete encrypted objects. Diagnose and authorize any further
capture separately.
