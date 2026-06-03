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

static void cpu_q3_ref(
    const std::vector<float>& ext_price,
    const std::vector<float>& discount,
    const std::vector<int>& l_shipdate_arr,
    const std::vector<int>& o_orderdate_arr,
    int cutoff_ymd,
    uint64_t& matched_count,
    uint64_t& sum_cents,
    double& sum_double
) {
  
    matched_count = 0;
    sum_cents = 0;
    sum_double = 0.0;
    const size_t Nlocal = l_shipdate_arr.size();

    for (size_t i = 0; i < Nlocal; i++) {
        if (o_orderdate_arr[i] < cutoff_ymd && l_shipdate_arr[i] > cutoff_ymd) {
            matched_count++;
            float  rev_f = ext_price[i] * (1.0f - discount[i]);
            double rev_d = double(ext_price[i]) * (1.0 - double(discount[i]));
            sum_cents += (uint64_t)(rev_f * 100.0f + 0.5f);
            sum_double += rev_d;
        }
    }

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

//-------------------------------------- MAIN-----------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
            << " customer.tbl orders.tbl lineitem.tbl [warmup] [reps]\n";
        return 1;
    }

    const std::string customer_path = argv[1];
    const std::string orders_path = argv[2];
    const std::string lineitem_path = argv[3];

    const int WARMUP = (argc > 4) ? std::atoi(argv[4]) : 30;
    const int REPS = (argc > 5) ? std::atoi(argv[5]) : 100;

    std::cout << "Loading customer from: " << customer_path << "\n";
    std::unordered_set<int32_t> building_custkeys;
    load_customer(customer_path, building_custkeys);
    auto join_t0 = std::chrono::high_resolution_clock::now();
    std::cout << "  Building-segment customers: " << building_custkeys.size() << "\n";

    std::cout << "Loading orders from: " << orders_path << "\n";
    std::unordered_map<int32_t, OrderInfo> order_map;
    load_orders_q3(orders_path, building_custkeys, order_map);
    std::cout << "  Qualifying orders: " << order_map.size() << "\n";
    building_custkeys.clear();

    std::cout << "Loading and pre-joining lineitem from: " << lineitem_path << "\n";
    std::vector<float>   ext_price, discount;
    std::vector<int>     l_shipdate_arr, o_orderdate_arr;
    int minShipYMD, maxShipYMD, minOrderYMD, maxOrderYMD;

    load_lineitem_q3(lineitem_path, order_map,
        ext_price, discount, l_shipdate_arr, o_orderdate_arr,
        minShipYMD, maxShipYMD, minOrderYMD, maxOrderYMD);
    order_map.clear();

    auto join_t1 = std::chrono::high_resolution_clock::now();  
    const double cpu_join_ms = std::chrono::duration<double, std::milli>(join_t1 - join_t0).count();
    std::cout << "  CPU join preprocessing time: " << cpu_join_ms << " ms\n"; 

    const uint32_t N = (uint32_t)l_shipdate_arr.size();
    std::cout << "  Pre-joined rows (N): " << N << "\n";
    std::cout << "  Shipdate range:      " << yyyymmdd_to_string(minShipYMD)
        << " to " << yyyymmdd_to_string(maxShipYMD) << "\n";
    std::cout << "  Orderdate range:     " << yyyymmdd_to_string(minOrderYMD)
        << " to " << yyyymmdd_to_string(maxOrderYMD) << "\n\n";


	// ---------- OpenCL setup ----------
    cl_platform_id platform;
    CHECK_CL(clGetPlatformIDs(1, &platform, nullptr));
    cl_device_id device;
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    char dev_name[256];
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr));
    std::cout << "Using device: " << dev_name << "\n";

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
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
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        std::cerr << "Build log:\n" << build_log.data() << "\n";
        CHECK_CL(err);
    }

    cl_kernel k = clCreateKernel(prog, "q3_aggregate", &err);
    CHECK_CL(err);

	// Create buffers and transfer data to GPU--------
    cl_mem d_ext_price = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, ext_price.data(), &err);    CHECK_CL(err);
    cl_mem d_discount = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, discount.data(), &err);     CHECK_CL(err);
    cl_mem d_l_shipdate = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * N, l_shipdate_arr.data(), &err); CHECK_CL(err);
    cl_mem d_o_orderdate = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * N, o_orderdate_arr.data(), &err); CHECK_CL(err);

    const size_t local = 256;
    const size_t global = ((size_t(N) + local - 1) / local) * local;
    const size_t num_groups = global / local;

    std::cout << "Workgroups: " << num_groups << " (local size: " << local << ")\n\n";

    cl_mem d_partials = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * num_groups, nullptr, &err); CHECK_CL(err);



	// Cutoff calculation

    std::vector<int> all_dates;
    all_dates.reserve(l_shipdate_arr.size() + o_orderdate_arr.size());
    for (int d : l_shipdate_arr)  all_dates.push_back(d);
    for (int d : o_orderdate_arr) all_dates.push_back(d);
    std::sort(all_dates.begin(), all_dates.end());
    all_dates.erase(std::unique(all_dates.begin(), all_dates.end()), all_dates.end());

    const int nc = (int)all_dates.size();
    std::cout << "Calibrating over all " << nc << " unique dates (" << N << " rows)...\n";
    std::vector<double> candidate_sel(nc, 0.0);
    for (int ci = 0; ci < nc; ci++) {
         int C = all_dates[ci];
         uint64_t cnt = 0;
         for (uint32_t i = 0; i < N; i++) {
              if (o_orderdate_arr[i] < C && l_shipdate_arr[i] > C) cnt++;
         }
         candidate_sel[ci] = double(cnt) / double(N);
    }

    // Find peak and restrict search to descending side (post-peak dates only)
    int peak_ci = (int)(std::max_element(candidate_sel.begin(),
        candidate_sel.end())
        - candidate_sel.begin());
    double peak_sel = candidate_sel[peak_ci];

    std::cout << "  Peak  selectivity: "
        << std::fixed << std::setprecision(4) << peak_sel * 100.0
        << "% at date " << yyyymmdd_to_string(all_dates[peak_ci]) << "\n";
    std::cout << "  Searching descending side only (dates after peak).\n";


    const std::vector<double> targets = {
        0.0025, 0.0050, 0.0075, 0.0100,
        0.0125, 0.0150, 0.0175, 0.0200,
        0.0225, 0.0250
    };

    std::vector<int> cutoffs;
    for (double target_s : targets) {
        int    best_ci = peak_ci;
        double best_diff = std::abs(candidate_sel[peak_ci] - target_s);
        for (int ci = peak_ci + 1; ci < nc; ci++) {
            double diff = std::abs(candidate_sel[ci] - target_s);
            if (diff < best_diff) { best_diff = diff; best_ci = ci; }
        }
        cutoffs.push_back(all_dates[best_ci]);
        std::cout << "  target=" << std::setprecision(4) << target_s * 100.0
            << "%  achieved=" << candidate_sel[best_ci] * 100.0
            << "%  cutoff=" << yyyymmdd_to_string(all_dates[best_ci]) << "\n";
    }
    std::cout << "Calibration done.\n\n";

	// CSV output setup
    std::ofstream csv("q3_results.csv");
    csv << "# cpu_join_preprocessing_ms=" << std::fixed << std::setprecision(3)
        << cpu_join_ms << ",pre_joined_rows=" << N << "\n";
    csv << "target_selectivity,cutoff_date,matched_rows,achieved_selectivity,"
        << "kernel_ms_min,kernel_ms_median,kernel_ms_max,"
        << "total_execution_time_median,cpu_join_preprocessing_ms,total_with_preprocessing_ms,"
        << "overhead_ms,overhead_percentage,"
        << "gpu_to_cpu_transfer_in_ms,cpu_reduction_time_in_ms,"
        << "useful_data_MB,total_data_MB,"
        << "estimated_bandwidth_GB_per_sec,"
        << "validation_cpu_result,validation_gpu_result,abs_err_cents,rel_err\n";
    csv << std::fixed << std::setprecision(9);

    struct TimingResult {
        double   kernel_ms;
        double   wall_ms;
        double   d2h_ms;
        double   cpu_finalize_ms;
        uint64_t gpu_sum_cents;
    };

    std::vector<uint64_t> partials(num_groups);

    auto launch_once = [&](int cutoff_ymd) -> TimingResult {
        TimingResult timing;
        auto wall_t0 = std::chrono::high_resolution_clock::now();

        CHECK_CL(clSetKernelArg(k, 0, sizeof(cl_mem), &d_ext_price));
        CHECK_CL(clSetKernelArg(k, 1, sizeof(cl_mem), &d_discount));
        CHECK_CL(clSetKernelArg(k, 2, sizeof(cl_mem), &d_l_shipdate));
        CHECK_CL(clSetKernelArg(k, 3, sizeof(cl_mem), &d_o_orderdate));
        CHECK_CL(clSetKernelArg(k, 4, sizeof(cl_mem), &d_partials));
        CHECK_CL(clSetKernelArg(k, 5, sizeof(int), &cutoff_ymd));
        CHECK_CL(clSetKernelArg(k, 6, sizeof(cl_uint), &N));

        cl_event evt;
        CHECK_CL(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, &evt));
        CHECK_CL(clFinish(q));

        cl_ulong t0 = 0, t1 = 0;
        CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
        CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr));
        timing.kernel_ms = double(t1 - t0) * 1e-6;
        clReleaseEvent(evt);

        auto d2h_t0 = std::chrono::high_resolution_clock::now();
        CHECK_CL(clEnqueueReadBuffer(q, d_partials, CL_TRUE, 0,
            sizeof(uint64_t) * num_groups, partials.data(), 0, nullptr, nullptr));
        auto d2h_t1 = std::chrono::high_resolution_clock::now();
        timing.d2h_ms = std::chrono::duration<double, std::milli>(d2h_t1 - d2h_t0).count();

        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        uint64_t sum_cents = 0;
        for (size_t i = 0; i < num_groups; i++) sum_cents += partials[i];
        timing.gpu_sum_cents = sum_cents;
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        timing.cpu_finalize_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

        auto wall_t1 = std::chrono::high_resolution_clock::now();
        timing.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
        return timing;
        };

    std::cout << "Starting Q3 selectivity sweep...\n";
    std::cout << "target_s | cutoff     | achieved_s | kernel_ms  | wall_ms   | overhead% | abs-err_cents | rel_err\n";
    std::cout << "---------------------------------------------------------------------------------------------------\n";

    for (size_t t_idx = 0; t_idx < targets.size(); t_idx++) {
        double target_s = targets[t_idx];
        int    cutoff_ymd = cutoffs[t_idx];

        uint64_t cpu_matched = 0, cpu_sum_cents = 0;
        double   cpu_revenue = 0.0;
        cpu_q3_ref(ext_price, discount, l_shipdate_arr, o_orderdate_arr,
            cutoff_ymd, cpu_matched, cpu_sum_cents, cpu_revenue);
        const double achieved_s = double(cpu_matched) / double(N);

        for (int i = 0; i < WARMUP; i++) { TimingResult dummy = launch_once(cutoff_ymd); (void)dummy; }

        std::vector<TimingResult> timings;
        timings.reserve(REPS);
        for (int r = 0; r < REPS; r++) {
            timings.push_back(launch_once(cutoff_ymd));
            sleep_ms(200);
        }

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

        const uint64_t gpu_sum_cents = filtered_wall[filtered_wall.size() / 2].gpu_sum_cents;
        const double   gpu_revenue = double(gpu_sum_cents) / 100.0;



		// Absolute error in cents to avoid floating point issues. Mirrors GPU kernel's fixed-point calculation.

        const uint64_t abs_err_cents = (cpu_sum_cents > gpu_sum_cents)
            ? cpu_sum_cents - gpu_sum_cents : gpu_sum_cents - cpu_sum_cents;

        // Relative difference between GPU (float) and CPU double-precision total.
        const double rel_err = (cpu_revenue > 0.0)
			? std::abs(cpu_revenue - gpu_revenue) / cpu_revenue : 0.0; // Avoid division by zero.

        const double row_size_bytes = sizeof(float) * 2 + sizeof(int) * 2;
        const double useful_data_MB = double(cpu_matched) * row_size_bytes / 1e6;
        const double total_data_MB = double(N) * row_size_bytes / 1e6;
        const double bytes_read = double(N) * row_size_bytes;
        const double bandwidth_GBps = bytes_read / (ms_med * 1e6);

        csv << target_s << "," << yyyymmdd_to_string(cutoff_ymd) << "," << cpu_matched << "," << achieved_s << ","
            << ms_min << "," << ms_med << "," << ms_max << ","
            << wall_ms_med << "," << cpu_join_ms << ","
            << (cpu_join_ms + wall_ms_med) << "," << overhead_ms << "," << overhead_pct << ","
            << d2h_ms_med << "," << cpu_finalize_ms_med << ","
            << useful_data_MB << "," << total_data_MB << ","
            << bandwidth_GBps << ","
            << cpu_revenue << "," << gpu_revenue << "," << abs_err_cents << "," << rel_err << "\n";

        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(8) << target_s << " | "
            << std::setw(10) << yyyymmdd_to_string(cutoff_ymd) << " | "
            << std::setw(10) << std::setprecision(6) << achieved_s << " | "
            << std::setw(10) << std::setprecision(3) << ms_med << " | "
            << std::setw(9) << wall_ms_med << " | "
            << std::setw(9) << std::setprecision(1) << overhead_pct << " | "
            << std::setw(9) << abs_err_cents << " | "
            << std::setprecision(6) << rel_err << "\n";
    }

csv.close();
std::cout << "\nWrote q3_results.csv\n";
// ---------- Cleanup ----------

clReleaseMemObject(d_ext_price);
clReleaseMemObject(d_discount);
clReleaseMemObject(d_l_shipdate);
clReleaseMemObject(d_o_orderdate);
clReleaseMemObject(d_partials);
clReleaseKernel(k);
clReleaseProgram(prog);
clReleaseCommandQueue(q);
clReleaseContext(ctx);

return 0;
}
