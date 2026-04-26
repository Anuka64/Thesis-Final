// TPC-H Q1

#define CL_TARGET_OPENCL_VERSION 120
#define NOMINMAX
#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#ifdef _WIN32
#include <windows.h>
#define sleep_ms(x) Sleep(x)
#else
#include <unistd.h>
#define sleep_ms(x) usleep((x)*1000)
#endif

#define CHECK_CL(err) \
    if ((err) != CL_SUCCESS) { \
        std::cerr << "OpenCL error " << (err) << " at line " << __LINE__ << "\n"; \
        std::exit(1); \
    }

static int parse_yyyymmdd(const std::string& s) {
    int y = std::stoi(s.substr(0, 4));
    int m = std::stoi(s.substr(5, 2));
    int d = std::stoi(s.substr(8, 2));
    return y * 10000 + m * 100 + d;
}

static std::string yyyymmdd_to_string(int yyyymmdd) {
    int y = yyyymmdd / 10000;
    int m = (yyyymmdd / 100) % 100;
    int d = yyyymmdd % 100;
    std::ostringstream oss;
    oss << y << "-";
    if (m < 10) oss << "0";
    oss << m << "-";
    if (d < 10) oss << "0";
    oss << d;
    return oss.str();
}
// Q1 Specifications ---------------
struct GroupKey {
    uint8_t rf, ls;
    bool operator<(const GroupKey& o) const {
        if (rf != o.rf) return rf < o.rf;
        return ls < o.ls;
    }
};

struct Aggregates {
    double sum_qty = 0.0;
    double sum_base_price = 0.0;
    double sum_disc_price = 0.0;
    double sum_charge = 0.0;
    double sum_discount = 0.0;
    uint64_t count_order = 0;
};



// data loading ------------------------
static void load_lineitem(
    const std::string& path,
    std::vector<float>& quantity,
    std::vector<float>& price,
    std::vector<float>& discount,
    std::vector<float>& tax,
    std::vector<uint8_t>& returnflag,
    std::vector<uint8_t>& linestatus,
    std::vector<int>& shipdate,
    int& minYMD,
    int& maxYMD
) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open: " << path << "\n";
        std::exit(1);
    }

    minYMD = std::numeric_limits<int>::max();
    maxYMD = std::numeric_limits<int>::min();

    std::string line;
    while (std::getline(in, line)) {
        int field = 0;
        size_t start = 0;
        float q = 0.0f, p = 0.0f, d = 0.0f, t = 0.0f;
        uint8_t rf = 0, ls = 0;
        std::string shipdate_str;

        while (true) {
            size_t pos = line.find('|', start);
            if (pos == std::string::npos) break;
            std::string f = line.substr(start, pos - start);

            if (field == 4) q = std::stof(f);
            else if (field == 5) p = std::stof(f);
            else if (field == 6) d = std::stof(f);
            else if (field == 7) t = std::stof(f);
            else if (field == 8) rf = f.empty() ? 0 : (uint8_t)f[0];
            else if (field == 9) ls = f.empty() ? 0 : (uint8_t)f[0];
            else if (field == 10) { shipdate_str = f; break; }

            field++;
            start = pos + 1;
        }

        if (shipdate_str.size() != 10) continue;

        int ymd = parse_yyyymmdd(shipdate_str);
        minYMD = std::min(minYMD, ymd);
        maxYMD = std::max(maxYMD, ymd);

        quantity.push_back(q);
        price.push_back(p);
        discount.push_back(d);
        tax.push_back(t);
        returnflag.push_back(rf);
        linestatus.push_back(ls);
        shipdate.push_back(ymd);
    }
}

// ---------------- CPU reference Q1

static void cpu_q1(
    const std::vector<float>& quantity,
    const std::vector<float>& price,
    const std::vector<float>& discount,
    const std::vector<float>& tax,
    const std::vector<uint8_t>& returnflag,
    const std::vector<uint8_t>& linestatus,
    const std::vector<int>& shipdate,
    int cutoff_ymd,
    std::map<GroupKey, Aggregates>& groups,
    uint64_t& matched_count)
{
    groups.clear();
    matched_count = 0;
    const size_t N = shipdate.size();

    for (size_t i = 0; i < N; i++) {
        if (shipdate[i] <= cutoff_ymd) {
            matched_count++;
            GroupKey k{ returnflag[i], linestatus[i] };

            float disc_price = price[i] * (1.0f - discount[i]);
            float charge = disc_price * (1.0f + tax[i]);

            groups[k].sum_qty += quantity[i];
            groups[k].sum_base_price += price[i];
            groups[k].sum_disc_price += disc_price;
            groups[k].sum_charge += charge;
            groups[k].sum_discount += discount[i];
            groups[k].count_order++;
        }
    }
}
static double compute_data_efficiency(uint64_t passing_rows, uint64_t total_rows) {
    if (total_rows == 0) return 0.0;
    return double(passing_rows) / double(total_rows);
}

// MEMORY REDUCTION KERNEL-----------------------
static const char* kernel_src = R"CLC(
inline uint group_index(uchar rf, uchar ls) {
    uint rf_idx = (rf == 'A') ? 0 : (rf == 'N') ? 1 : (rf == 'O') ? 2 : 3;
    uint ls_idx = (ls == 'F') ? 0 : 1;
    return rf_idx * 4 + ls_idx;
}

__kernel void q1_aggregate(
    __global const float* quantity,
    __global const float* price,
    __global const float* discount,
    __global const float* tax,
    __global const uchar* returnflag,
    __global const uchar* linestatus,
    __global const int* shipdate,
    __global ulong* out_partials_qty,
    __global ulong* out__partials_base,
    __global ulong* out__partials_disc,
    __global ulong* out__partials_charge,
    __global ulong* out__partials_discount,
    __global uint* out__partials_count,
    __global uint* out__partials_matched,
    const int cutoff_ymd,
    const uint N
){
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group = get_group_id(0);
    uint lsize = get_local_size(0);

    const uint MAX_GROUPS = 16;
    
    __local ulong lsum_qty[16][64];
    __local ulong lsum_base[16][64];
    __local ulong lsum_disc[16][64];
    __local ulong lsum_charge[16][64];
    __local ulong lsum_discount[16][64];
    __local uint lsum_count[16][64];
    __local uint lsum_matched[64];

    // Initialize local memory
    for (uint g = 0; g < MAX_GROUPS; g++) {
        lsum_qty[g][lid] = 0;
        lsum_base[g][lid] = 0;
        lsum_disc[g][lid] = 0;
        lsum_charge[g][lid] = 0;
        lsum_discount[g][lid] = 0;
        lsum_count[g][lid] = 0;
    }
    lsum_matched[lid] = 0;

    // Process data
    if (gid < N) {
        int sd = shipdate[gid];
        if (sd <= cutoff_ymd) {
            lsum_matched[lid] = 1;
            
            float q = quantity[gid];
            float p = price[gid];
            float d = discount[gid];
            float t = tax[gid];
            uchar rf = returnflag[gid];
            uchar ls = linestatus[gid];
            
            float disc_price = p * (1.0f - d);
            float charge = disc_price * (1.0f + t);
            
            ulong qty_cents = (ulong)(q * 100.0f);
            ulong base_cents = (ulong)(p * 100.0f);
            ulong disc_cents = (ulong)(disc_price * 100.0f);
            ulong charge_cents = (ulong)(charge * 100.0f);
            ulong discount_cents = (ulong)(d * 100.0f);
            
            uint gidx = group_index(rf, ls);
            
            lsum_qty[gidx][lid] = qty_cents;
            lsum_base[gidx][lid] = base_cents;
            lsum_disc[gidx][lid] = disc_cents;
            lsum_charge[gidx][lid] = charge_cents;
            lsum_discount[gidx][lid] = discount_cents;
            lsum_count[gidx][lid] = 1;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (uint g = 0; g < MAX_GROUPS; g++) {
                lsum_qty[g][lid] += lsum_qty[g][lid + stride];
                lsum_base[g][lid] += lsum_base[g][lid + stride];
                lsum_disc[g][lid] += lsum_disc[g][lid + stride];
                lsum_charge[g][lid] += lsum_charge[g][lid + stride];
                lsum_discount[g][lid] += lsum_discount[g][lid + stride];
                lsum_count[g][lid] += lsum_count[g][lid + stride];
            }
            lsum_matched[lid] += lsum_matched[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partials
    if (lid == 0) {
        for (uint g = 0; g < MAX_GROUPS; g++) {
            out_partials_qty[group * MAX_GROUPS + g] = lsum_qty[g][0];
            out_partials_base[group * MAX_GROUPS + g] = lsum_base[g][0];
            out_partials_disc[group * MAX_GROUPS + g] = lsum_disc[g][0];
            out_partials_charge[group * MAX_GROUPS + g] = lsum_charge[g][0];
            out_partials_discount[group * MAX_GROUPS + g] = lsum_discount[g][0];
            out_partials_count[group * MAX_GROUPS + g] = lsum_count[g][0];
        }
        out_partials_matched[group] = lsum_matched[0];
    }
}
)CLC";


// --------------------------Main----------------------------
int main(int argc, char** argv) 
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " lineitem.tbl\n";
        return 1;
    }
    const std::string path = argv[1];
    const int WARMUP = (argc > 2) ? std::atoi(argv[2]) : 20;
    const int REPS = (argc > 3) ? std::atoi(argv[3]) : 50;

    std::cout << "=== TPC-H Query 1 - Step 7 ===\n";
    std::cout << "Loading lineitem from: " << path << "\n";
    
    std::vector<float> quantity, price, discount, tax;
    std::vector<uint8_t> returnflag, linestatus;
    std::vector<int> shipdate;
    int minYMD, maxYMD;
    
    load_lineitem(argv[1], quantity, price, discount, tax, returnflag, linestatus, shipdate, minYMD, maxYMD);

    const uint32_t N = (uint32_t)shipdate.size();
    std::cout << "Loaded " << shipdate.size() << " rows\n";
    std::cout << "Date range: " << yyyymmdd_to_string(minYMD)
        << " to " << yyyymmdd_to_string(maxYMD) << "\n\n";

	//------------- OpenCL Setup---------------------
    cl_platform_id platform;
    CHECK_CL(clGetPlatformIDs(1, &platform, nullptr));

    cl_device_id device;
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    char dev_name[256];
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr));
    std::cout << "Using device: " << dev_name << "\n";

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!ctx) { /* handle error, e.g. print err */ }

    cl_command_queue q = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err);

    const char* src_ptr = kernel_src;
    size_t src_len = std::strlen(kernel_src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    CHECK_CL(err);

    err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << "\n";
        CHECK_CL(err);
    }

    cl_kernel k = clCreateKernel(prog, "q1_aggregate", &err);
    CHECK_CL(err);


	//----- ----- Create buffers ----------
    
    cl_mem d_quantity = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, quantity.data(), &err);
    CHECK_CL(err);

    cl_mem d_price = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, price.data(), &err);
    CHECK_CL(err);

    cl_mem d_discount = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, discount.data(), &err);
    CHECK_CL(err);

    cl_mem d_tax = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, tax.data(), &err);
    CHECK_CL(err);

    cl_mem d_returnflag = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint8_t) * N, returnflag.data(), &err);
    CHECK_CL(err);

    cl_mem d_linestatus = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint8_t) * N, linestatus.data(), &err);
    CHECK_CL(err);

    cl_mem d_shipdate = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * N, shipdate.data(), &err);
    CHECK_CL(err);



	// Workgroup size
	const size_t local = 64; //Reduced to 64 to better fit the GPU's shared memory and reduce idle threads
    const size_t global = ((N + local - 1) / local) * local;
    const size_t num_groups = global / local;
    const uint32_t MAX_GROUPS = 16;
    
    std::cout << "Workgroups: " << num_groups << " (local size: " << local << ")\n";

    cl_mem d_partials_qty = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * num_groups * MAX_GROUPS, nullptr, &err);
    CHECK_CL(err);

    cl_mem d_partials_base = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * num_groups * MAX_GROUPS, nullptr, &err);
    CHECK_CL(err);

    cl_mem d_partials_disc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * num_groups * MAX_GROUPS, nullptr, &err);
    CHECK_CL(err);

    cl_mem d_partials_charge = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * num_groups * MAX_GROUPS, nullptr, &err);
    CHECK_CL(err);

    cl_mem d_partials_discount = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * num_groups * MAX_GROUPS, nullptr, &err);
    CHECK_CL(err);

    cl_mem d_partials_count = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * num_groups * MAX_GROUPS, nullptr, &err);
    CHECK_CL(err);

    cl_mem d_partials_matched = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * num_groups, nullptr, &err);
    CHECK_CL(err);

    // Sort shipdate to generate cutoffs
    std::vector<int> shipdate_sorted = shipdate;
    std::sort(shipdate_sorted.begin(), shipdate_sorted.end());

    // Selectivity targets
    const std::vector<double> targets = { 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0 };

    std::vector<int> cutoffs;
    for (double s : targets) {
        int idx = int(s * double(N));
        if (idx >= (int)N) idx = (int)N - 1;
        cutoffs.push_back(shipdate_sorted[idx]);
    }

    // CSV output --------------------
    std::ofstream csv("q1_results.csv");
    csv << "target_selectivity,cutoff_date,achieved_selectivity,"
        << "kernel_ms_min,kernel_ms_median,kernel_ms_max,"
        << "total_execution_time_median,overhead_ms,overhead_percentage,"
        << "gpu_to_cpu_transfer_in_ms,cpu_reduction_time_in_ms,"
        << "thread_utilization_percentage,wasted_threads_percentage,"
        << "data_efficiency_percentage,useful_data_MB,total_data_MB,"
        << "bandwidth_GB_per_sec,"
        << "num_groups,validation_cpu_result,validation_gpu_result,abs_error\n";
    csv << std::fixed << std::setprecision(9);

    // timing struct -----------------
    struct TimingResult {
        double kernel_ms;
        double wall_ms;
        double d2h_ms;
        double cpu_finalize_ms;
        std::vector<uint64_t> sum_qty, sum_base, sum_disc, sum_charge, sum_discount;
        std::vector<uint32_t> count;
        uint32_t matched;
        double total_charge;
        uint32_t num_groups;
    };
   
    // -- Launch helper. 
    auto launch_once = [&](int cutoff_ymd) -> TimingResult {
        TimingResult timing;

        auto wall_t0 = std::chrono::high_resolution_clock::now();

        CHECK_CL(clSetKernelArg(k, 0, sizeof(cl_mem), &d_quantity));
        CHECK_CL(clSetKernelArg(k, 1, sizeof(cl_mem), &d_price));
        CHECK_CL(clSetKernelArg(k, 2, sizeof(cl_mem), &d_discount));
        CHECK_CL(clSetKernelArg(k, 3, sizeof(cl_mem), &d_tax));
        CHECK_CL(clSetKernelArg(k, 4, sizeof(cl_mem), &d_returnflag));
        CHECK_CL(clSetKernelArg(k, 5, sizeof(cl_mem), &d_linestatus));
        CHECK_CL(clSetKernelArg(k, 6, sizeof(cl_mem), &d_shipdate));
        CHECK_CL(clSetKernelArg(k, 7, sizeof(cl_mem), &d_partials_qty));
        CHECK_CL(clSetKernelArg(k, 8, sizeof(cl_mem), &d_partials_base));
        CHECK_CL(clSetKernelArg(k, 9, sizeof(cl_mem), &d_partials_disc));
        CHECK_CL(clSetKernelArg(k, 10, sizeof(cl_mem), &d_partials_charge));
        CHECK_CL(clSetKernelArg(k, 11, sizeof(cl_mem), &d_partials_discount));
        CHECK_CL(clSetKernelArg(k, 12, sizeof(cl_mem), &d_partials_count));
        CHECK_CL(clSetKernelArg(k, 13, sizeof(cl_mem), &d_partials_matched));
        CHECK_CL(clSetKernelArg(k, 14, sizeof(int), &cutoff_ymd));
        CHECK_CL(clSetKernelArg(k, 15, sizeof(cl_uint), &N));

        cl_event evt;
        CHECK_CL(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, &evt));
        CHECK_CL(clFinish(q));

        cl_ulong t0 = 0, t1 = 0;
        CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
        CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr));
        timing.kernel_ms = double(t1 - t0) * 1e-6;
        clReleaseEvent(evt);

        // GPU to CPU transfer
        auto d2h_t0 = std::chrono::high_resolution_clock::now();

        std::vector<uint64_t> partials_qty(num_groups* MAX_GROUPS);
        std::vector<uint64_t> partials_base(num_groups* MAX_GROUPS);
        std::vector<uint64_t> partials_disc(num_groups* MAX_GROUPS);
        std::vector<uint64_t> partials_charge(num_groups* MAX_GROUPS);
        std::vector<uint64_t> partials_discount(num_groups* MAX_GROUPS);
        std::vector<uint32_t> partials_count(num_groups* MAX_GROUPS);
        std::vector<uint32_t> partials_matched(num_groups);

        CHECK_CL(clEnqueueReadBuffer(q, d_partials_qty, CL_TRUE, 0, sizeof(uint64_t)* num_groups* MAX_GROUPS,
            partials_qty.data(), 0, nullptr, nullptr));
        CHECK_CL(clEnqueueReadBuffer(q, d_partials_base, CL_TRUE, 0, sizeof(uint64_t)* num_groups* MAX_GROUPS,
            partials_base.data(), 0, nullptr, nullptr));
        CHECK_CL(clEnqueueReadBuffer(q, d_partials_disc, CL_TRUE, 0, sizeof(uint64_t)* num_groups* MAX_GROUPS,
            partials_disc.data(), 0, nullptr, nullptr));
        CHECK_CL(clEnqueueReadBuffer(q, d_partials_charge, CL_TRUE, 0, sizeof(uint64_t)* num_groups* MAX_GROUPS,
            partials_charge.data(), 0, nullptr, nullptr));
        CHECK_CL(clEnqueueReadBuffer(q, d_partials_discount, CL_TRUE, 0, sizeof(uint64_t)* num_groups* MAX_GROUPS,
            partials_discount.data(), 0, nullptr, nullptr));
        CHECK_CL(clEnqueueReadBuffer(q, d_partials_count, CL_TRUE, 0, sizeof(uint32_t)* num_groups* MAX_GROUPS,
            partials_count.data(), 0, nullptr, nullptr));
        CHECK_CL(clEnqueueReadBuffer(q, d_partials_matched, CL_TRUE, 0, sizeof(uint32_t)* num_groups,
            partials_matched.data(), 0, nullptr, nullptr));

        auto d2h_t1 = std::chrono::high_resolution_clock::now();
        timing.d2h_ms = std::chrono::duration<double, std::milli>(d2h_t1 - d2h_t0).count();

        // combine results from all workgroups into final group sum
        auto cpu_t0 = std::chrono::high_resolution_clock::now();

        timing.sum_qty.resize(MAX_GROUPS, 0);
        timing.sum_base.resize(MAX_GROUPS, 0);
        timing.sum_disc.resize(MAX_GROUPS, 0);
        timing.sum_charge.resize(MAX_GROUPS, 0);
        timing.sum_discount.resize(MAX_GROUPS, 0);
        timing.count.resize(MAX_GROUPS, 0);
        timing.matched = 0;

        for (size_t wg = 0; wg < num_groups; wg++) {
            timing.matched += partials_matched[wg];
            for (uint32_t g = 0; g < MAX_GROUPS; g++) {
                size_t idx = wg * MAX_GROUPS + g;
                timing.sum_qty[g] += partials_qty[idx];
                timing.sum_base[g] += partials_base[idx];
                timing.sum_disc[g] += partials_disc[idx];
                timing.sum_charge[g] += partials_charge[idx];
                timing.sum_discount[g] += partials_discount[idx];
                timing.count[g] += partials_count[idx];
            }
        }

        timing.total_charge = 0.0;
        timing.num_groups = 0;
        for (uint32_t g = 0; g < MAX_GROUPS; g++) {
            if (timing.count[g] > 0) {
                timing.num_groups++;
                timing.total_charge += double(timing.sum_charge[g]) / 100.0;
            }
        }

        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        timing.cpu_finalize_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

        auto wall_t1 = std::chrono::high_resolution_clock::now();
        timing.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
        return timing;
        };


    // Selectivity sweep
    std::cout << "Starting Q1 selectivity sweep...\n";
    std::cout << "target_s | cutoff     | achieved_s | kernel_ms  | wall_ms   | overhead% | groups | abs_err\n";
    std::cout << "--------------------------------------------------------------------------------------------\n";

    for (size_t t_idx = 0; t_idx < targets.size(); t_idx++) {
        double target_s = targets[t_idx];
        int cutoff_ymd = cutoffs[t_idx];

    // CPU Reference
    std::map<GroupKey, Aggregates> cpu_groups;
    uint64_t cpu_matched = 0;
    cpu_q1(quantity, price, discount, tax, returnflag, linestatus, shipdate,
        cutoff, cpu_groups, cpu_matched);

    const double achieved_s = double(cpu_matched) / double(N);
    for (int i = 0; i < WARMUP; i++) { TimingResult dummy = launch_once(cutoff_ymd); (void)dummy; }

    std::vector<TimingResult> timings;
    timings.reserve(REPS);
    for (int r = 0; r < REPS; r++) { timings.push_back(launch_once(cutoff_ymd)); sleep_ms(200); }

    std::sort(timings.begin(), timings.end(), [](const TimingResult& a, const TimingResult& b) {
        return a.kernel_ms < b.kernel_ms; });

    const size_t remove_count = timings.size() * 35 / 100;
    std::vector<TimingResult> filtered_kernel;
    for (size_t i = remove_count; i < timings.size() - remove_count; i++)
        filtered_kernel.push_back(timings[i]);

    const double ms_min = filtered_kernel.front().kernel_ms;
    const double ms_max = filtered_kernel.back().kernel_ms;
    const double ms_med = filtered_kernel[filtered_kernel.size() / 2].kernel_ms;

    std::vector<TimingResult> timings_by_wall = timings;
    std::sort(timings_by_wall.begin(), timings_by_wall.end(), [](const TimingResult& a, const TimingResult& b) {
        return a.wall_ms < b.wall_ms; });
    std::vector<TimingResult> filtered_wall;
    for (size_t i = remove_count; i < timings_by_wall.size() - remove_count; i++)
        filtered_wall.push_back(timings_by_wall[i]);

    const double wall_ms_med = filtered_wall[filtered_wall.size() / 2].wall_ms;
    const double d2h_ms_med = filtered_wall[filtered_wall.size() / 2].d2h_ms;
    const double cpu_finalize_ms_med = filtered_wall[filtered_wall.size() / 2].cpu_finalize_ms;
    const double overhead_ms = wall_ms_med - ms_med;
    const double overhead_pct = (overhead_ms / wall_ms_med) * 100.0;

    const TimingResult& median_run = filtered_wall[filtered_wall.size() / 2];
    const uint32_t num_groups_result = median_run.num_groups;
    const double total_gpu_charge = median_run.total_charge;
    
    // Calculate CPU total
    const char returnflags[] = { 'A', 'N', 'O', 'R' };
    const char linestatuses[] = { 'F', 'O', ' ', ' ' };
    double total_cpu_charge = 0.0;

    for (uint32_t g = 0; g < MAX_GROUPS; g++) {
        if (median_run.count[g] > 0) {
            uint8_t rf = returnflags[g / 4];
            uint8_t ls = linestatuses[g % 4];
            GroupKey k{ rf, ls };

            if (cpu_groups.count(k) > 0) {
                total_cpu_charge += cpu_groups[k].sum_charge;
            }
        }
    }

    // Metrics
    const double abs_err = std::abs(total_cpu_charge - total_gpu_charge);
    const double thread_util_pct = achieved_s * 100.0;
    const double wasted_threads_pct = (1.0 - achieved_s) * 100.0;
    const double row_size_bytes = sizeof(float) * 4 + sizeof(uint8_t) * 2 + sizeof(int);
    const double useful_data_MB = double(cpu_matched) * row_size_bytes / 1e6;
    const double total_data_MB = double(N) * row_size_bytes / 1e6;
    const double data_efficiency_pct = compute_data_efficiency(cpu_matched, N) * 100.0;
    const double bytes_read = double(N) * row_size_bytes;
    const double bytes_written = double(num_groups) * MAX_GROUPS * (sizeof(uint64_t) * 5 + sizeof(uint32_t)) +
        double(num_groups) * sizeof(uint32_t);
    const double total_bytes = bytes_read + bytes_written;
    const double theoritical_gbps_med = total_bytes / (ms_med * 1e6);

    // Write CSV
    csv << target_s << "," << yyyymmdd_to_string(cutoff_ymd) << "," << achieved_s << ","
        << ms_min << "," << ms_med << "," << ms_max << ","
        << wall_ms_med << "," << overhead_ms << "," << overhead_pct << ","
        << d2h_ms_med << "," << cpu_finalize_ms_med << ","
        << thread_util_pct << "," << wasted_threads_pct << ","
        << data_efficiency_pct << "," << useful_data_MB << "," << total_data_MB << ","
        << theoritical_gbps_med << ","
        << num_groups << "," << total_cpu_charge << "," << total_gpu_charge << "," << abs_err << "\n";

    // Console output
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(8) << target_s << " | "
        << std::setw(10) << yyyymmdd_to_string(cutoff_ymd) << " | "
        << std::setw(10) << std::setprecision(6) << achieved_s << " | "
        << std::setw(10) << std::setprecision(3) << ms_med << " | "
        << std::setw(9) << wall_ms_med << " | "
        << std::setw(9) << std::setprecision(1) << overhead_pct << " | "
        << std::setw(6) << num_groups << " | "
        << std::setprecision(3) << abs_err << "\n";
    }

    csv.close();
    std::cout << "\nWrote q1_results.csv\n";
    // ---------- Cleanup ----------
    clReleaseMemObject(d_quantity); 
    clReleaseMemObject(d_price);
    clReleaseMemObject(d_discount);
    clReleaseMemObject(d_tax);
    clReleaseMemObject(d_returnflag);
    clReleaseMemObject(d_linestatus);
    clReleaseMemObject(d_shipdate);
    clReleaseMemObject(d_partials_qty);
    clReleaseMemObject(d_partials_base);
    clReleaseMemObject(d_partials_disc);
    clReleaseMemObject(d_partials_charge);
    clReleaseMemObject(d_partials_discount);
    clReleaseMemObject(d_partials_count);
    clReleaseMemObject(d_partials_matched);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return 0;
}
