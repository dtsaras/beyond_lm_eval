# Round 15, partial (23 of 51) — preserved copy

gpt-6-astra hit its usage limit mid-pass. These 23 verdicts landed;
the other 28 failed in 0.3 min with the quota error.

Tally: 2 GREEN / 5 YELLOW / 16 RED.

Adjudicated so far:
  geometry_cka, geometry_collapse  CONFIRMED, and the cause was a regression
    introduced by my own dtype-preserving fix to collect_hidden_states: the
    offload writes raw bytes while the memmap read hard-coded float32, so
    float64 wrote 8-byte floats read back as 4-byte. FIXED (np_dtype recorded
    in the offload metadata and reused by the reader).
  geometry_hubness                 NOT REPRODUCED on any fixture I can build,
    including astra's own 4-point one, across single- and multi-batch paths.
    Filed as needing their fixture.

Still to adjudicate: the remaining ~13 REDs.

These are preserved because the watcher quarantines live verdicts when it wakes.
