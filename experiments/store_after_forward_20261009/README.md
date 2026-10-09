# Noncritical stores after forward

The scoped-fence trace measured about 28–31 ms per episode of D2H copy API
work exposed between compute batches. The original worker submits every
deferred store before forward. This proposal retains early submission and
fences for reuse-critical jobs, then submits known-producer noncritical jobs
only after the next forward has been submitted. Current-step stores retain
the original deferral to the next step.

This builds on specific producer events so moving submission does not create
a dependency on the newer forward. Missing producers, load-only steps, unsafe
KV mutations and reset drains use the ordinary path. Completion acknowledgments
still require real DMA completion on every rank. The flag is disabled by default.

Compare against producer events alone, with all other serving settings fixed.
Audit the new `after_forward` markers and repeat-level throughput and GPU gaps.
