### **Parallelization Strategies Comparison**

| **Type**                    | **Scope**                  | **How It Works**                                                                                         | **Example**                                                                |
| --------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **ILP (Instruction-Level)** | Within a single core       | Executes multiple instructions from the same thread in parallel (via pipelining, superscalar execution). | Modern CPUs using out-of-order execution and pipelining.                   |
| **TLP (Thread-Level)**      | Across threads             | Multiple threads run concurrently, sharing resources or working independently.                           | Multithreaded apps (e.g., Java threads, POSIX threads).                    |
| **DLP (Data-Level)**        | Across data elements       | Same operation applied to multiple data points simultaneously.                                           | SIMD instructions, GPU kernels, vectorized loops in NumPy.                 |
| **Task-Level**              | Across tasks/functions     | Different tasks execute in parallel, often asynchronously.                                               | Web servers handling multiple requests; parallel pipelines in ML training. |
| **Pipeline Parallelism**    | Across stages of a process | Breaks computation into stages; each stage processes different data concurrently.                        | CPU instruction pipeline; deep learning model pipeline parallelism.        |
| **Process-Level**           | Across processes           | Multiple processes run in parallel, often on different cores or machines.                                | MPI in HPC clusters; multi-process Python using `multiprocessing`.         |
| **GPU Parallelism**         | Massive data parallelism   | Thousands of lightweight threads execute on GPU cores for data-heavy tasks.                              | CUDA/OpenCL for deep learning, image processing.                           |
| **Distributed Parallelism** | Across multiple nodes      | Workload distributed across multiple machines connected via network.                                     | Hadoop MapReduce, Apache Spark, Kubernetes clusters.                       |

***

✅ **Key Observations:**

*   **ILP** is hardware-driven and invisible to programmers.
*   **TLP, DLP, Task-Level** are software-visible and often require explicit programming.
*   **GPU & Distributed Parallelism** scale massively but need specialized frameworks.
