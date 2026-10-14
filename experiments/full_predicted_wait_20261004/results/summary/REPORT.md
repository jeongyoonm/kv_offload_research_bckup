# Full predicted wait: GLM 5.3 TP8

Completed cells: 48/48. Accepted: 48.

Values are arithmetic means across available seed 57/58 cells. Gains use the
matched capped control. Both phases are complete. See paired_comparisons.json
for individual seed comparisons; two replays do not establish statistical
significance.

<!-- markdownlint-disable MD013 -->

| Workload | Policy | Repeats | req/s | Gain | Mean TTFT ms | p95 TTFT ms | Compute rounds | Accounted polling CPU ms | Exposed hold ms | Sleep after stop ms | Admission ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p61440_f95_h100 | capped50_fine2 | 2 | 6.947 | +0.00% | 4007.9 | 5769.1 | 13.0 | 79.19 | 399.33 | 2.60 | 95.1% |
| p61440_f95_h100 | full_margin0_fine10 | 2 | 7.031 | +1.21% | 3982.0 | 5853.8 | 14.0 | 154.44 | 334.21 | 16.45 | 91.5% |
| p61440_f95_h100 | full_margin0_fine2 | 2 | 7.034 | +1.25% | 3958.3 | 5837.5 | 14.0 | 155.30 | 317.69 | 4.53 | 92.3% |
| p61440_f95_h100 | full_margin0_fine5 | 2 | 7.034 | +1.25% | 3980.1 | 5816.8 | 14.0 | 175.50 | 322.10 | 5.36 | 95.1% |
| p61440_f95_h100 | full_margin50_fine10 | 2 | 7.022 | +1.07% | 3946.7 | 5384.8 | 13.0 | 202.83 | 366.44 | 11.46 | 91.8% |
| p61440_f95_h100 | full_margin50_fine2 | 2 | 7.082 | +1.94% | 3921.7 | 5577.6 | 13.5 | 171.16 | 311.25 | 5.64 | 91.0% |
| p61440_f95_h100 | full_margin50_fine5 | 2 | 7.056 | +1.56% | 3940.5 | 5645.2 | 13.5 | 156.15 | 364.76 | 5.60 | 93.1% |
| p61440_f95_h100 | readiness2 | 2 | 7.009 | +0.89% | 3956.6 | 5413.8 | 13.0 | 170.54 | 312.21 | 4.43 | 93.8% |
| p61440_f99_h100 | capped50_fine2 | 2 | 11.556 | +0.00% | 2509.8 | 3561.4 | 13.0 | 94.09 | 388.01 | 7.23 | 100.0% |
| p61440_f99_h100 | full_margin0_fine10 | 2 | 11.504 | -0.45% | 2545.7 | 3459.0 | 13.0 | 185.84 | 414.66 | 13.77 | 99.3% |
| p61440_f99_h100 | full_margin0_fine2 | 2 | 11.536 | -0.17% | 2495.2 | 3683.9 | 13.5 | 167.42 | 340.94 | 7.79 | 100.0% |
| p61440_f99_h100 | full_margin0_fine5 | 2 | 11.583 | +0.24% | 2501.3 | 3692.5 | 13.0 | 186.69 | 344.90 | 15.22 | 100.0% |
| p61440_f99_h100 | full_margin50_fine10 | 2 | 11.641 | +0.74% | 2488.6 | 3679.5 | 13.0 | 141.51 | 372.20 | 21.55 | 100.0% |
| p61440_f99_h100 | full_margin50_fine2 | 2 | 11.439 | -1.01% | 2544.7 | 3632.2 | 13.0 | 248.09 | 333.62 | 4.37 | 100.0% |
| p61440_f99_h100 | full_margin50_fine5 | 2 | 11.431 | -1.08% | 2543.5 | 3600.0 | 13.0 | 139.00 | 415.27 | 5.65 | 100.0% |
| p61440_f99_h100 | readiness2 | 2 | 11.613 | +0.49% | 2491.1 | 3648.9 | 13.0 | 121.91 | 335.79 | 6.22 | 100.0% |
| p8192_f95_h100 | capped50_fine2 | 2 | 41.775 | +0.00% | 736.1 | 949.2 | 38.5 | 61.86 | 68.00 | 1.69 | 94.8% |
| p8192_f95_h100 | full_margin0_fine10 | 2 | 42.507 | +1.75% | 727.0 | 916.7 | 38.0 | 195.62 | 77.02 | 2.66 | 91.6% |
| p8192_f95_h100 | full_margin0_fine2 | 2 | 41.540 | -0.56% | 741.0 | 1023.4 | 39.0 | 214.36 | 77.47 | 2.61 | 92.7% |
| p8192_f95_h100 | full_margin0_fine5 | 2 | 42.273 | +1.19% | 732.9 | 998.7 | 38.0 | 206.71 | 88.43 | 2.66 | 91.6% |
| p8192_f95_h100 | full_margin50_fine10 | 2 | 41.568 | -0.50% | 743.2 | 962.7 | 38.5 | 187.04 | 59.03 | 2.08 | 93.3% |
| p8192_f95_h100 | full_margin50_fine2 | 2 | 42.071 | +0.71% | 730.0 | 984.7 | 38.5 | 214.96 | 75.36 | 1.34 | 94.4% |
| p8192_f95_h100 | full_margin50_fine5 | 2 | 41.525 | -0.60% | 744.8 | 1113.2 | 40.0 | 193.76 | 10.67 | 0.61 | 94.3% |
| p8192_f95_h100 | readiness2 | 2 | 41.973 | +0.47% | 735.2 | 978.1 | 38.0 | 80.50 | 107.53 | 2.47 | 90.7% |

<!-- markdownlint-enable MD013 -->

## Timing and CPU limits

Sleep-after-stop is the sum of recorded wait intervals after an eligible stop
condition and preceding compute completion. Candidate ACK times are engine
observations; completion timing uses its upper endpoint. It is a diagnostic
estimate, not an exact causal latency bound.

Main polling CPU covers iterations leading to adaptive waits (and observer
installation), excluding final/pre-ACK iterations. Bridge CPU covers the entire
observer thread. Their sum is accounted polling CPU, not whole-engine CPU.
Unknown GPU timing windows are excluded and listed in cells.json. The bridge
polls read-only transport state every 1 ms during coarse waits.
