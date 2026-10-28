# Model development continuity

- Before a new calibration, read `docs/model-development-journal.md` and the
  compatible entry in `data/model-registry/development-index.json`.
- Carry compatible learned parameter seeds forward. Do not reset to generic
  generator defaults after a solver change. A new core MUST recompute scores
  and state; a prior parameter seed is not a cached trajectory.
- Keep numerical-method evidence, material/chemistry laws, and fixture-specific
  gas geometry separate. Never apply a fitted gas route as a universal material
  coefficient or transfer it to another layout without explicit qualification.
- Conditioned post-burst diagnostics are NOT full-from-zero calibration or
  independent validation. Candidate profiles stay candidates until both the
  relevant numerical checks and experimental boundary checks have passed.
- Preserve raw experiments, archived native binaries, best profiles and failed
  prefixes. Record provenance and checks in the journal. No silent rollback or
  undocumented parameter/core reset.
- Minimum physical step and strict mass/chemical/thermal/oxygen/UI ledgers are
  constraints, not optimization variables. Reusable numerical fixes need an
  analytical/isolated regression plus a real CUDA replay before promotion.
- FSRI search must qualify the collective burst BEFORE preferring cooling:
  use explosion-priority policy, reject missing cascade or peak/front timing
  errors >100 s, target <=10 s. A better tail cannot compensate for a bad burst.
  Older cooling-only records must be reassessed, not assumed burst-qualified.
