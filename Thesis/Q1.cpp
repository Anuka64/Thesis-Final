
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
static void load_lineitem_needed(
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

        float p = 0.0f, d = 0.0f;
        std::string shipdate_str;

        while (true) {
            size_t pos = line.find('|', start);
            if (pos == std::string::npos) break;
            std::string f = line.substr(start, pos - start);

            if (field == 5) p = std::stof(f);
            else if (field == 6) d = std::stof(f);
            else if (field == 10) { shipdate_str = f; break; }

            field++;
            start = pos + 1;
        }

        if (shipdate_str.size() != 10) continue;

        int ymd = parse_yyyymmdd(shipdate_str);
        minYMD = std::min(minYMD, ymd);
        maxYMD = std::max(maxYMD, ymd);

        int di = days_since_epoch(parse_ymd(shipdate_str));
        price.push_back(p);
        discount.push_back(d);
        ship_day.push_back(di);
    }
}

// ---------------- CPU reference Q1

static double cpu_q1(
    const std::vector<float>& price,
    const std::vector<float>& discount,
    const std::vector<int>& shipdate,
    int lo_day,
    int hi_day
) {
    double sum = 0.0;
    const size_t N = price.size();
    for (size_t i = 0; i < N; i++) {
        if (ship_day[i] >= lo_day && ship_day[i] < hi_day &&
            discount[i] >= 0.05f && discount[i] <= 0.07f) {
            sum += double(price[i]) * double(discount[i]);
        }
    }
    return sum;
}


// OpenCL kernel from Q6-----------------------
static const char* kernel_src = R"CLC(
__kernel void q6_kernel_reduce(
    __global const float* price,
    __global const float* discount,
    __global const int* ship_day,
    __global ulong* out_partials,
    const int lo_day,
    const int hi_day,
    const uint N
){
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group = get_group_id(0);
    uint lsize = get_local_size(0);

    __local ulong lsum[256]; // requires local size <= 256
    ulong x = 0;

   if (gid < N) {
        int sd = ship_day[gid];
        float disc = discount[gid];
        if (sd >= lo_day && sd < hi_day &&
            disc >= 0.05f && disc <= 0.07f) {
            float y = price[gid] * disc;
            // Convert to fixed-point cents for accuracy
            ulong cents = (ulong)(y * 100.0f);
            x = cents;
        }
    }

    lsum[lid] = x;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            lsum[lid] += lsum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out_partials[group] = lsum[0];
    }
}
)CLC";


// --------------------------Main----------------------------
int main(int argc, char** argv) 
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " lineitem.tbl\n";
        return 1;
    }

    const std::string path = argv[1];

    // ---------- Load data ----------
    std::vector<float> quantity, price, discount, tax;
    std::vector<uint8_t> returnflag, linestatus;
    std::vector<int> shipdate;
    int minYMD, maxYMD;
    load_lineitem(argv[1], quantity, price, discount, tax, returnflag, linestatus, shipdate, minYMD, maxYMD);

    std::cout << "Loaded " << shipdate.size() << " rows\n";
    std::cout << "Date range: " << yyyymmdd_to_string(minYMD)
        << " to " << yyyymmdd_to_string(maxYMD) << "\n";
    return 0;
}
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
