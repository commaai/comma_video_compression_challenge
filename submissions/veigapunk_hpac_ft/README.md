# veigapunk_hpac_ft

Fine-tuned semantic renderer on top of `semantic-pose-HPAC_CPR1` (PR #130).

- Same HPAC token stream and pose carrier (CPR1)
- 800-step AdamW fine-tune of the int4 semantic renderer against SegNet CE at eval resolution
- Archive: 191,028 bytes (−24 B vs CPR1 191,052)

## Lineage
Derived from PR #130 / fesalfayed CPR1 (semantic maps + pose carrier + integer HPAC).
