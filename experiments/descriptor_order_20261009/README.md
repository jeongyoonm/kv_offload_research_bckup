# Preserve transfer descriptor order

Coalescing previously sorted descriptors even when no ranges could merge.
The private block-major CPU layout has one aggregate KV row per cache block,
so per-layer rows are separated by all other layers. Sorting alone does not
reduce its descriptor count and changes the GPU transfer access sequence.

`descriptor_preserve_order` is opt-in and disabled by default. Without merges
it leaves the original arrays unchanged. With merges, it orders each merged
group by its first original descriptor index. Byte coverage and addresses are
unchanged. It applies to both load and store descriptors.

The idle-GPU matrix uses the actual private mmap implementations and all 99
GLM page geometries, ascending and descending GPU assignments, and disabled,
legacy sorted, and preserved coalescing. Each case checks every loaded byte,
untouched rows, and exactly-once native retirement. The matrix identifies
transfer path cost; it does not identify a particular GPU hardware mechanism.

CPU verification: 10 targeted helper tests passed. GPU and serving findings
will be recorded after the serialized validation gate completes.
