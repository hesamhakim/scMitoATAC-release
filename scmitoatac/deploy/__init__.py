"""scmitoatac.deploy — Phase-7 deployment layer (partition-first, abstention-honest, relative-preferred).

Built ON TOP of the FROZEN caller (scmitoatac v0.1.0, see docs/deployment/CALLER_FREEZE.md).
This layer does NOT touch frozen scoring math; it orchestrates the caller across
partition-defined metacells and licenses claims against the characterized envelope.

Pipeline (master plan v3.5, Phase 7 §§7-D.0-7-D.5):
  Step 0  partition   : CONSUMED as a label column in .obs (not reimplemented here).
  Step 1  feasibility : feasibility.py  -- per-compartment depth-floor calculator.
  Step 2  call        : metacell.py     -- pool cells by partition, call via frozen caller,
                                           abstain per achieved depth.
  Step 3  relative    : nullcompare.py  -- between-compartment differences vs the random-pool null.
  Step 4  licensing    : licensing.py    -- claim licensing + used_in_partition double-dipping guard.

Status: SCAFFOLD. feasibility (Step 1) is complete and tested; metacell/nullcompare/licensing
carry working cores wired to the frozen package with AnnData I/O marked where it plugs in.
AnnData/MuData are imported lazily (optional `deploy` extra) so the arithmetic core runs without them.
"""
from . import feasibility  # noqa
from . import partition  # noqa  -- Step 0 boundary: standard PartitionResult + registry + synthetic gens
from . import adapters  # noqa  -- external tool adapters (register on import)

__all__ = ["adapters", "feasibility", "partition", "metacell", "nullcompare", "licensing"]
