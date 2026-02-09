
#Repo structure

#Thesis (main codes) 
-microbenchmark 
----Operator level OpenCL microbenchmarks (scan+filter, Projection+compaction, scaler aggregation, group-by aggregation)--Runs on synthetic data, not actual TPC-H data. (To isolate from all the dependencies and measure GPU kernel performance)

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

# Setup for the entire experiment
- Install Visual Studio with C++ support (2017 or later)
- Download and install OpenCl SDK (I had Intel SDK for OpenCl Applications)
- Install vtTune (Used to measure GPU execution on my OpenCL kernel)
- Insatll python (Used for plotiing the measurement output. )


# Dependencies
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
 
# Command to run and produce plots from spripts 

- Install python library : pip install pandas matplotlib (from comman promt)
- Put all the CSV result and plot scripts in the same file then run in cmd : python 'script_name'.py  (e.g python plot_microbenchmark.py)


      
