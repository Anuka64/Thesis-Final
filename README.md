Thesis Masters Thesis

#Repo structure

#Thesis (main codes) |   
-microbenchmark 
----Operator level OpenCL microbenchmarks (scan+filter, Projection+compaction, scaler aggregation, group-by aggregation)

-Q6 
---Full implementation of TPC-H query 6, including CPU reference for correctness   
-Q1 
--- OpenCL implementation of TCP-H query 1 (GPU kernel only)

#Scripts |   
- Build, run, and profiling instructions (python + VTune)
- Data generation scripts (TPC-H dbgen)
- Result collection and plotting scripts
- CSV files for collected results

#Plots     
- Plots generated from the collected results in the experimentsS