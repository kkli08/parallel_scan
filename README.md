# parallel_scan

```shell
*******************************************************************************************************
Total time: 9.638464 ms
Timing breakdown:
        Time to copy input data: 1.987712 ms
        Time in small_inclusive_scan_kernel: 0.013312 ms
        Time in block_inclusive_scan_kernel: 5.101856 ms
        Time in adjust_with_block_sums_kernel: 1.992544 ms
        Time in recursive calls: 0.235520 ms
        Time in memory allocations: 0.114656 ms
*******************************************************************************************************
```

```shell
*******************************************************************************************************
Performance Results:
        Time consumed by the sequential implementation: 152052us
        Time consumed by my implementation: 10124us
        Optimization Speedup Ratio (nearest integer): 15
*******************************************************************************************************
```