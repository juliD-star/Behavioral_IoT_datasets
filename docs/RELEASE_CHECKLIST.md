# Prepublication checklist

- [ ] Compare all released class supports, capture counts, metrics, and confusion matrices against final manuscript Tables III, VI, VII, S1 and Figure S1.
- [ ] Re-run the supplied ML script in a clean environment; compare saved metrics within documented tolerances.
- [ ] Check raw PCAPs already on GitHub for MAC/IP addresses, hostnames, tokens, account metadata, and third-party traffic; encrypted payloads do not guarantee anonymity.
- [ ] Review CSV categorical fields, metadata, scripts, and saved result files for sensitive information and embedded filesystem paths.
- [ ] Verify original authorship, co-author approval, ownership and an appropriate license.
- [ ] Confirm the paper's precise feature definitions, especially mean inter-arrival time units.
- [ ] Review large files, GitHub size limits, and repository history before committing.
- [ ] Upload contents to the **existing** repository without deleting its v1.0 PCAP data.
- [ ] Tag `v1.1.0` and publish a GitHub release; verify Zenodo archives the **new** version and obtain its new DOI.
- [ ] Update manuscript Data Availability and reviewer response only after the new DOI is accessible.
