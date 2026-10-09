# Critical store submission priority

Branch: `experiment/store-critical-submission-20261009`.
Parent: `6aeab52`.
Artifacts: `/mnt/localnvme/jeongyoon/sp1`.

D2H jobs are submitted in FIFO order on one ordered transfer lane. Stores whose
GPU source blocks are needed for reuse can wait behind unrelated unsubmitted
stores. Prioritize only those unsubmitted jobs referenced by final metadata's
flush set, using stable ordering within each priority class. The CPU backend
first verifies that all destination block sets are disjoint; overlapping or
unknown destination specs preserve the original order. Already submitted jobs,
source-producer stream dependencies, completion events, and all-rank retirement
are unchanged. Both arms use the existing global GPU store fence. Defaults off.
