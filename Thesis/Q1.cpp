
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

            if (field == 4) p = std::stof(f);
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

static double cpu_q1(
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
    __global ulong* out_qty,
    __global ulong* out_base,
    __global ulong* out_disc,
    __global ulong* out_charge,
    __global ulong* out_discount,
    __global uint* out_count,
    __global uint* out_matched,
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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " lineitem.tbl\n";
        return 1;
    }

    // ---------- Load data ----------
    std::vector<float> quantity, price, discount, tax;
    std::vector<uint8_t> returnflag, linestatus;
    std::vector<int> shipdate;
    int minYMD, maxYMD;
    load_lineitem(argv[1], quantity, price, discount, tax, returnflag, linestatus, shipdate, minYMD, maxYMD);

    const uint32_t N = (uint32_t)shipdate.size()
    std::cout << "Loaded " << shipdate.size() << " rows\n";
    std::cout << "Date range: " << yyyymmdd_to_string(minYMD)
        << " to " << yyyymmdd_to_string(maxYMD) << "\n\n";

	//------------- New OpenCL part---------------------
    cl_platform_id platform;
    CHECK_CL(clGetPlatformIDs(1, &platform, nullptr));

    cl_device_id device;
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    char dev_name[256];
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr));
    std::cout << "Using device: " << dev_name << "\n";

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr &err);
	CHECK_CL(err);

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


    // Build kernel
    const char* src_ptr = kernel_src;
    size_t src_len = strlen(kernel_src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, nullptr);
    clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel k = clCreateKernel(prog, "q1_aggregate", nullptr);

	//----- ----- Create buffers ----------
    cl_int err;
    cl_mem d_shipdate = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * N, shipdate.data(), &err);
    CHECK_CL(err);

    cl_mem d_matched = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t), nullptr, &err);
    CHECK_CL(err);

    // Partial results buffers
    const size_t local = 64;
    const size_t global = ((N + local - 1) / local) * local;
    const size_t num_groups = global / local;
    const uint32_t MAX_GROUPS = 16;

    cl_mem d_partials_qty = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * num_groups * MAX_GROUPS, nullptr, &err);
    CHECK_CL(err);

    // ---------- Cleanup ----------
    clReleaseMemObject(d_price);
    clReleaseMemObject(d_discount);
    clReleaseMemObject(d_shipday);
    clReleaseMemObject(d_partials);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return 0;
}
