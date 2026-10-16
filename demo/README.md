# Demo documents

Three contracts from CUAD's **official test split** (The Atticus Project, CC-BY-4.0), chosen
by length (≈ the split's 35th percentile, median×2, and p90) and for having many clause
types present, so the defence demo has something to show on every question:

| file | CUAD title | chars | present clause types |
|---|---|---|---|
| `short_supply_agreement.txt` | AgapeAtpCorp_20191202_10-KA_EX-10.1_11911128_EX-10.1_Supply Agreement | 20,693 | 19 |
| `medium_sponsorship_agreement.txt` | N2KINC_10_16_1997-EX-10.16-SPONSORSHIP AGREEMENT | 49,518 | 18 |
| `long_outsourcing_agreement.txt` | TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING AGREEMENT | 101,639 | 19 |

`demo_documents.json` carries CUAD's gold spans for each, so the demo can be checked
against ground truth live. The PDFs with real page numbers are in CUAD's full release
(`CUAD_v1.zip` → `full_contract_pdf/`, same titles); the `.txt` files here are the exact
text the model was evaluated on. Pre-extracted caches for the three ship in `demo/cache/`
once M1 is done, so the demo never waits on extraction.
