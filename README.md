Topic: Characterizing TPC-H Selectivity Effects on Intel Iris GPU Performance for Analytical Query Processing

#Research Questions
RQ1
How does predicate selectivity affect scan and filter execution on Intel integrated GPUs, in terms of occupancy, divergence, and memory bandwidth utilization?
RQ2
How do selectivity changes propagate from scan–filter operators to end-to-end performance in filter-dominated TPC-H queries on Intel integrated GPUs?
RQ3
How does selectivity affect aggregation-heavy analytical queries on Intel integrated GPUs, and which hardware bottlenecks dominate at high selectivity?
RQ4
Based on the observed selectivity-dependent behavior of filter- and aggregation-heavy queries, what selectivity-aware optimization insights can be derived?

#The goal of this experiment is to demonstrate how selectivity effects iGPU performance at:
Operator level (scan, filter, aggregation)
Query level (Q6 and Q1)

All experiments were conducted on a fixed hardware and software configuration.
#Hardware Environment: 
| Component   | Specification                           |
| ----------- | --------------------------------------- |
| CPU         | 11th Gen Intel Core i7-1165G7 @ 2.80GHz |
| GPU         | Intel Iris Xe Graphics                  |
| GPU Driver  | 32.0.101.7026                           |
| Driver Date | 19.08.2025                              |
| RAM         | 8 GB                                    |
| Storage     | SSD – WDC PC SN530 512GB                |
#Hardware level (Intel Iris Xe integrated GPU)
#Operating System:  Windows 11  Version: 25H2  Build: 26200.7840

#Software Environment
| Software              | Version                               |
| --------------------- | ------------------------------------- |
| Visual Studio         | 2026 (18.1.1)                         |
| MSVC Toolset          | v145                                  |
| C++ Standard          | ISO C++23 Preview (/std:c++23preview) |
| OpenCL Headers        | Khronos (CL/cl.h)                     |
| OpenCL Implementation | Intel oneAPI OpenCL 2025.3            |
| OpenCL Platform       | Intel(R) OpenCL Graphics              |
| oneAPI                | 2025.3                                |
| Intel VTune           | 2025.7                                |
| Python                | 3.11.7                                |
| pandas                | 2.3.3                                 |
| matplotlib            | 3.10.8                                |

#Dataset Generation : Benchmark Version: TPC-H Version 3.0.1
Sacle Factor: SF= 1. Future: SF=5 and 10

#Generate Data:  1. Download TPC-H v3.0.1 ->2 Navigate to dbgen directory
3. Build dbgen: go to cd C:\TPC-HV3.0.1\dbgen or where dbgen folder is -> open Visual Studio cmd, run : "nmake /f makefile.suite" -> It should generate dbgen.exe file. 
    for SF=1 run "type dbgen -s 1" in the same cmd or -s 5 for SF=5 
    Here, I mainly used lineitem.tbl file. 
    
#Repo structure

#Thesis (main codes) 
-microbenchmark 
----Operator level OpenCL microbenchmarks (scan+filter, Projection+compaction, scaler aggregation, group-by aggregation)--Runs on synthetic data, not actual TPC-H data. (To isolate from all the dependencies and measure GPU kernel performance)

-Q6 
---Full implementation of TPC-H query 6, including CPU reference for correctness   
-Q1 
--- OpenCL implementation of TCP-H query 1 (GPU kernel only)

#Scripts |   
- Result collection and plotting scripts
- CSV files for collected results

#Plots     
- Plots generated from the collected results in the experimentsS

# Setup for the entire experiment
- Install Visual Studio with C++ support (2017 or later)
- Download and install OpenCl SDK (I had Intel SDK for OpenCl Applications)
- Install vtTune (Used to measure GPU execution on my OpenCL kernel)
- Insatll python (Used for plotiing the measurement output. )


# Dependencies setup in Visual studio
- OpenCl header (CL/cl.h)
    - Go to visual studio -> Solution Explorer -> Properties -> On top (configuration -> Release, Platform-> Active 64)
    - Go to Configuration Properties -> C/c++ -> General -> Aditional Include Directories (put the address for the cl) : C:\CL\CL-Header
- OpenCl library (OpenCL.lib)
    - Go to visual studio -> Solution Explorer -> Properties -> On top (configuration -> Release, Platform-> Active 64)
    - Go to Configuration Properties -> Linker -> General -> Additional Library Directorries C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\lib (I had a probmel with OpenCl library so used OneAPI library. Nothing fundamentally changed because of it. In normal cases, it should point where you OpenCl library file is)
    - go to Input -> Additional Dependencies : OpenCL.lib
- Set the location of the data Source
    - Go to visual studio -> Solution Explorer -> Properties -> Configuration Properties -> Debugging -> Command Arguments : C:\TPC-H-V3.0.1\dbgen\lineitem.tbl (location of your data table)
    - General -> Configuration type : Application (.exe)
-Visual Studio Configuration
    - Configuration: Release
    - Platform: x64

#Running the experiments:
1. Open VS -> create new project -> Console App -> Next
2. Put the codes from repo in the location and open them: Solution Explorer -> Source Files -> right click -> Add -> Existing item -> Select the code you want to run.
3. set all the dependencies.
4. Build -> Build Solution
5. Debug -> Start without debugging.

#Output :
Will run the code. Show info in the cmd
Generate CSV file containing result. 
keep the result in the same folder where ploting script is. 

# Command to run and produce plots from spripts 

- Install python library : pip install pandas matplotlib (from comman promt)
- Put all the CSV result and plot scripts in the same file then run in cmd : python 'script_name'.py  'csv_file_name.csv' (e.g python plot_microbenchmark.py result.csv)

# Experiment Protocol
To ensure reproducibility:
-Close all background applications
-Set Windows power mode to Best Performance
-Reboot system before benchmark session
-Use Release build only
-Run each experiment in isolation


      
