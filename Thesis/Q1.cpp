
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


// OpenCL kernel from v1-----------------------
static const char* kernel_src = R"CLC(
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
    if (gid >= N) return;

    if (shipdate[gid] <= cutoff_ymd) {
        float q = quantity[gid];
        float p = price[gid];
        float d = discount[gid];
        float t = tax[gid];
        uchar rf = returnflag[gid];
        uchar ls = linestatus[gid];

        float disc_price = p * (1.0f - d);
        float charge = disc_price * (1.0f + t);
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

    std::cout << "Loaded " << shipdate.size() << " rows\n";
	//------------- New OpenCL part---------------------
    cl_platform_id platform;
    CHECK_CL(clGetPlatformIDs(1, &platform, nullptr));

    cl_device_id device;
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, nullptr);

    // Build kernel
    const char* src_ptr = kernel_src;
    size_t src_len = strlen(kernel_src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, nullptr);
    clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel k = clCreateKernel(prog, "q1_aggregate", nullptr);
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
