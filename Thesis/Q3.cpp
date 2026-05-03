#define CL_TARGET_OPENCL_VERSION 120
#define NOMINMAX
#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
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

static double compute_data_efficiency(uint64_t passing_rows, uint64_t total_rows) {
    if (total_rows == 0) return 0.0;
    return double(passing_rows) / double(total_rows);
}

struct OrderInfo {
    int orderdate;
    int shippriority;
};

// Step 1: collect custkeys for the BUILDING market segment.
static void load_customer(
    const std::string& path,
    std::unordered_set<int32_t>& building_custkeys
) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }

    std::string line;
    while (std::getline(in, line)) {
        int field = 0;
        size_t start = 0;
        int32_t custkey = 0;
        std::string mkt;

        while (true) {
            size_t pos = line.find('|', start);
            if (pos == std::string::npos) break;
            std::string f = line.substr(start, pos - start);
            if (field == 0) custkey = std::stoi(f);
            else if (field == 6) { mkt = f; break; }
            field++;
            start = pos + 1;
        }
        if (mkt == "BUILDING")
            building_custkeys.insert(custkey);
    }
}

// Step 2: load orders which belongs to BUILDING-segment of customers.
static void load_orders_q3(
    const std::string& path,
    const std::unordered_set<int32_t>& building_custkeys,
    std::unordered_map<int32_t, OrderInfo>& order_map
) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }

    std::string line;
    while (std::getline(in, line)) {
        int field = 0;
        size_t start = 0;
        int32_t orderkey = 0, custkey = 0;
        std::string orderdate_str;
        int shippriority = 0;

        while (true) {
            size_t pos = line.find('|', start);
            if (pos == std::string::npos) break;
            std::string f = line.substr(start, pos - start);
            if (field == 0)      orderkey = std::stoi(f);
            else if (field == 1) custkey = std::stoi(f);
            else if (field == 4) orderdate_str = f;
            else if (field == 7) { shippriority = std::stoi(f); break; }
            field++;
            start = pos + 1;
        }

        if (orderdate_str.size() == 10 && building_custkeys.count(custkey))
            order_map[orderkey] = { parse_yyyymmdd(orderdate_str), shippriority };
    }
}

// Step 3: join lineitem against the order map and write in GPU.
static void load_lineitem_q3(
    const std::string& path,
    const std::unordered_map<int32_t, OrderInfo>& order_map,
    std::vector<float>& ext_price,
    std::vector<float>& discount,
    std::vector<int>& l_shipdate_arr,
    std::vector<int>& o_orderdate_arr,
    int& minShipYMD, int& maxShipYMD,
    int& minOrderYMD, int& maxOrderYMD
) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }

    minShipYMD = std::numeric_limits<int>::max();
    maxShipYMD = std::numeric_limits<int>::min();
    minOrderYMD = std::numeric_limits<int>::max();
    maxOrderYMD = std::numeric_limits<int>::min();

    std::string line;
    while (std::getline(in, line)) {
        int field = 0;
        size_t start = 0;
        int32_t ok = 0;
        float ep = 0.0f, disc = 0.0f;
        std::string shipdate_str;

        while (true) {
            size_t pos = line.find('|', start);
            if (pos == std::string::npos) break;
            std::string f = line.substr(start, pos - start);
            if (field == 0)       ok = std::stoi(f);
            else if (field == 5)  ep = std::stof(f);
            else if (field == 6)  disc = std::stof(f);
            else if (field == 10) { shipdate_str = f; break; }
            field++;
            start = pos + 1;
        }

        if (shipdate_str.size() != 10) continue;

        auto it = order_map.find(ok);
        if (it == order_map.end()) continue;

        int sd = parse_yyyymmdd(shipdate_str);
        const OrderInfo& oi = it->second;

        minShipYMD = std::min(minShipYMD, sd);
        maxShipYMD = std::max(maxShipYMD, sd);
        minOrderYMD = std::min(minOrderYMD, oi.orderdate);
        maxOrderYMD = std::max(maxOrderYMD, oi.orderdate);

        ext_price.push_back(ep);
        discount.push_back(disc);
        l_shipdate_arr.push_back(sd);
        o_orderdate_arr.push_back(oi.orderdate);
    }
}

// CPU reference implementation for validation.

static double cpu_q3(
    const std::vector<float>& ext_price,
    const std::vector<float>& discount,
    const std::vector<int>& l_shipdate_arr,
    const std::vector<int>& o_orderdate_arr,
    const std::vector<int32_t>& orderkey_arr,
    const std::vector<int32_t>& shippriority_arr,
    int cutoff_ymd,
    uint64_t& matched_count
) {
    std::map<Q3GroupKey, double> groups;
    matched_count = 0;
    const size_t Nlocal = l_shipdate_arr.size();

    for (size_t i = 0; i < Nlocal; i++) {
        if (o_orderdate_arr[i] < cutoff_ymd && l_shipdate_arr[i] > cutoff_ymd) {
            matched_count++;
            double revenue = double(ext_price[i]) * (1.0 - double(discount[i]));
            Q3GroupKey k{ orderkey_arr[i], o_orderdate_arr[i], shippriority_arr[i] };
            groups[k] += revenue;
        }
    }

    double total = 0.0;
    for (const auto& kv : groups) total += kv.second;
    return total;
}
// Mirrors GPU kernels decimal point exactly.
static uint64_t cpu_q3_fixed(
    const std::vector<float>& ext_price,
    const std::vector<float>& discount,
    const std::vector<int>& l_shipdate_arr,
    const std::vector<int>& o_orderdate_arr,
    int cutoff_ymd
) {
    uint64_t sum_cents = 0;
    const size_t Nlocal = l_shipdate_arr.size();
    for (size_t i = 0; i < Nlocal; i++) {
        if (o_orderdate_arr[i] < cutoff_ymd && l_shipdate_arr[i] > cutoff_ymd) {
            float rev = ext_price[i] * (1.0f - discount[i]);
            sum_cents += (uint64_t)(rev * 100.0f + 0.5f);
        }
    }
    return sum_cents;
}

// openCl Kernel

static const char* kernel_src = R"CLC(
__kernel void q3_aggregate(
    __global const float* ext_price,
    __global const float* discount,
    __global const int*   l_shipdate,
    __global const int*   o_orderdate,
    __global ulong*       out_partials,
    const int  cutoff_ymd,
    const uint N
) {
    uint gid   = get_global_id(0);
    uint lid   = get_local_id(0);
    uint group = get_group_id(0);
    uint lsize = get_local_size(0);

    __local ulong lsum[256];
    ulong x = 0;

    if (gid < N) {
        int sd = l_shipdate[gid];
        int od = o_orderdate[gid];
        if (od < cutoff_ymd && sd > cutoff_ymd) {
            float rev = ext_price[gid] * (1.0f - discount[gid]);
            x = (ulong)(rev * 100.0f + 0.5f);
        }
    }

    lsum[lid] = x;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) lsum[lid] += lsum[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) out_partials[group] = lsum[0];
}
)CLC";

csv.close();
std::cout << "\nWrote q3_results.csv\n";
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
