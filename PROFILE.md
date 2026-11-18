# PROFILING RESULTS
This markdown file contains the summary for the profiling results done by utilities `nys`
using Nsight systems package

## KERNEL SUMMARY

| Time (%) | Total Time (ns) | Instances | Kernel Name |
| :---: | :---: | :---: | :--- |
| **18.7** | 1,344,982 | 142 | `BackwardCLWeight` |
| **13.5** | 972,130 | 142 | `BackwardPLBias` |
| **12.0** | 861,573 | 142 | `BackwardCLBias` |
| **7.7** | 554,925 | 142 | `BackwardPLWeight` |
| **7.6** | 545,361 | 142 | `ForwardFCLPreActivate` |
| **6.7** | 481,521 | 426 | `ApplyActivate` |
| **6.3** | 452,208 | 426 | `ApplyGradient` |
| **2.8** | 200,124 | 142 | `BackwardCLOutput` |
| **2.5** | 183,196 | 142 | `BackwardCLPreActivate` |
| **2.5** | 178,677 | 142 | `ForwardCLPreActivate` |
| **2.4** | 176,535 | 142 | `ForwardPLPreActivate` |
| **2.4** | 175,996 | 142 | `BackwardPLPreActivate` |
| **2.3** | 164,953 | 142 | `BackwardPLOutput` |
| **2.2** | 161,850 | 142 | `ForwardCLBias` |
| **2.2** | 158,780 | 142 | `BackwardFCLWeight` |
| **2.1** | 152,377 | 142 | `ForwardFCLBias` |
| **2.1** | 149,274 | 142 | `ForwardPLBias` |
| **2.1** | 147,870 | 142 | `BackwardFCLBias` |
| **2.0** | 145,213 | 142 | `ComputeError` |

## MEMORY SUMMARY
| Time (%) | Total Time (ns) | Count | Operation |
| :---: | :---: | :---: | :--- |
| **95.2** | 2,090,414 | 2,704 | `[CUDA memset]` |
| **4.8** | 105,502 | 148 | `[CUDA memcpy Host-to-Device]` |
