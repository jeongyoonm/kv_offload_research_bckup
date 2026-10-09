# Preencoded client control

A producer-event repeat developed a 473 ms load-only stall before the first
changed store dependency. The original first 32 client starts span about 80 ms.
This control encodes identical request bodies before measurement and releases
the first wave through a client barrier. Subsequent requests retain the same
closed-loop executor and concurrency.

Both arms use the new client path. Compare paired arms within this allocation;
do not attribute differences from historical harness runs to server changes.
Frozen predictor, prefixes, suffixes, model execution and server flags match.
Every measured cell retains cache-hit, output-token, preemption and preparation
validation. Report first-wave start spread as a manipulation check.
