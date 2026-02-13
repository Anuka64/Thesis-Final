// TPC-H Q6 standard
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

//  Date utilities (Gregorian calendar)
struct YMD { int y, m, d; };

static YMD parse_ymd(const std::string& s) {
    // "1996-03-13"
    return { std::stoi(s.substr(0,4)), std::stoi(s.substr(5,2)), std::stoi(s.substr(8,2)) };
}

static bool is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

static int days_since_epoch(YMD dt) {
    static const int mdays_norm[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };
    int days = 0;
    for (int y = 1970; y < dt.y; y++) days += is_leap(y) ? 366 : 365;
    for (int m = 1; m < dt.m; m++) {
        days += mdays_norm[m - 1];
        if (m == 2 && is_leap(dt.y)) days += 1;
    }
    days += (dt.d - 1);
    return days;
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


// Load lineitem.tbl
// Loads:
// l_quantity       (float)
//  - l_extendedprice (float)
//  - l_discount      (float)
//  - l_shipdate      (int day index since 1970-01-01)
// Also tracks min/max shipdate as YYYYMMDD for printing.
static void load_lineitem_needed(
    const std::string& path,
    std::vector<float>& quantity,
    std::vector<float>& price,
    std::vector<float>& discount,
    std::vector<int>& ship_day,
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

        float q = 0.0f, p = 0.0f, d = 0.0f;
        std::string shipdate_str;

        while (true) {
            size_t pos = line.find('|', start);
            if (pos == std::string::npos) break;
            std::string f = line.substr(start, pos - start);

            if (field == 4) q = std::stof(f);
            else if (field == 5) p = std::stof(f);
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
		quantity.push_back(q);
        price.push_back(p);
        discount.push_back(d);
        ship_day.push_back(di);
    }
}


// ---------------- CPU reference Q6 

static double cpu_q6(
	const std::vector<float>& quantity,
    const std::vector<float>& price,
    const std::vector<float>& discount,
    const std::vector<int>& ship_day,
    int lo_day,
    int hi_day
) {
    double sum = 0.0;
    const size_t N = price.size();
    for (size_t i = 0; i < N; i++) {
        if (ship_day[i] >= lo_day && ship_day[i] < hi_day &&
            discount[i] >= 0.03f && discount[i] <= 0.09f &&
            quantity[i]< 28.0f){
            sum += double(price[i]) * double(discount[i]);
        }
    }
    return sum;
}
// --- Helper for data efficiency calculation

static double compute_data_efficiency(uint64_t passing_rows, uint64_t total_rows) {
    if (total_rows == 0) return 0.0;
    return double(passing_rows) / double(total_rows);
}
static uint64_t cpu_q6_count(
	const std::vector<float>& quantity,
    const std::vector<int>& ship_day,
    const std::vector<float>& discount,
    int lo_day,
    int hi_day
) {
    uint64_t cnt = 0;
    const size_t N = ship_day.size();
    for (size_t i = 0; i < N; i++) {
        if (ship_day[i] >= lo_day && ship_day[i] < hi_day &&
            discount[i] >= 0.03f && discount[i] <= 0.09f &&
            quantity[i] < 28.0f) {
            cnt++;
        }
    }
    return cnt;
}

// OpenCL kernel
static const char* kernel_src = R"CLC(
__kernel void q6_kernel_reduce(
    __global const float* quantity,
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
        float qty = quantity[gid];
        if (sd >= lo_day && sd < hi_day &&
            disc >= 0.03f && disc <= 0.09f &&
            qty < 28.0f) {
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


// ---------------- Device selection, CPU fallback ---------------

static cl_device_id pick_device(cl_platform_id platform) {
    cl_device_id dev = nullptr;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    if (err == CL_SUCCESS) return dev;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, nullptr);
    CHECK_CL(err);
    return dev;
}



//  selectivity W for Q6 targets 

static int calibrate_W_for_target(
	const std::vector<float>& quantity,
    const std::vector<int>& ship_day_sorted,
    const std::vector<float>& discount_sorted, // <-- add this parameter
    int D0_day,
    double target_s
) {
    const size_t N = ship_day_sorted.size();
	if (N == 0) return -1;

    size_t max_idx = ship_day_sorted.size() - 1;
    if (D0_day > ship_day_sorted[max_idx]) {
		return -1;
    }

    auto geq_it = std::lower_bound(ship_day_sorted.begin(), ship_day_sorted.end(), D0_day);
    size_t start_idx = std::distance(ship_day_sorted.begin(), geq_it);

	uint64_t target_passing = (uint64_t)(target_s * double(N) + 0.001);
	if (target_passing == 0) target_passing = 1;

	uint64_t cnt_so_far = 0;
	int best_W = -1;    

    for (size_t i = start_idx; i < N; i++) {
        int current_end_day = ship_day_sorted[i];
        if (discount_sorted[i] >= 0.03f && discount_sorted[i] <= 0.09f &&
            quantity[i] < 28.0f) {  
            cnt_so_far++;
            if (cnt_so_far >= target_passing) {
                best_W = current_end_day - D0_day + 1;
                break;
            }
        }
    }
    if (best_W < 0) {
        if (cnt_so_far > 0) {
            best_W = ship_day_sorted.back() - D0_day + 1;
        }
        else {
            best_W = -1;
        }
    }

    return best_W;
}

// =============================================================
// Main
// =============================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <lineitem.tbl>\n";
        return 1;
    }

    // ---------- Load data ----------
    std::vector<float> quantity, price, discount;
    std::vector<int> ship_day;
    int minYMD, maxYMD;
    load_lineitem_needed(argv[1], quantity, price, discount, ship_day, minYMD, maxYMD);

    const uint32_t N = static_cast<uint32_t>(price.size());
    std::cout << "Rows (N): " << N << "\n";
    std::cout << "Min shipdate: " << yyyymmdd_to_string(minYMD) << "\n";
    std::cout << "Max shipdate: " << yyyymmdd_to_string(maxYMD) << "\n";

    //Q6 anchor
    const std::string D0_str = "1994-01-01";
    const int D0_day = days_since_epoch(parse_ymd(D0_str));
    std::cout << "Q6 anchor D0: " << D0_str << "\n\n";

    // Build shipdate for calibration 
    std::vector<size_t> indices(N);
    for (size_t i = 0; i < N; i++) {
        indices[i] = i;
	}
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return ship_day[a] < ship_day[b];
		}); 
    std::vector<float> quantity_sorted(N);  
    std::vector<int> ship_day_sorted(N);
    std::vector<float> discount_sorted(N);
    for (size_t i = 0; i < N; i++) {
        quantity_sorted[i] = quantity[indices[i]];  
        ship_day_sorted[i] = ship_day[indices[i]];
        discount_sorted[i] = discount[indices[i]];
    }

    const std::vector<double> targets = {0.5}; 

    // ---------- OpenCL setup ----------
    cl_platform_id platform;
    CHECK_CL(clGetPlatformIDs(1, &platform, nullptr));
    cl_device_id device = pick_device(platform);
    cl_int err;

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err);

    cl_command_queue q = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, nullptr, &err);
    CHECK_CL(err);

    err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t sz = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
        std::vector<char> log(sz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sz, log.data(), nullptr);
        std::cerr << log.data() << "\n";
        return 1;
    }

    cl_kernel k = clCreateKernel(prog, "q6_kernel_reduce", &err);
    CHECK_CL(err);

    //Buffer
    cl_mem d_quantity = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * (size_t)N, quantity.data(), &err);
    CHECK_CL(err);

    cl_mem d_price = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * (size_t)N, price.data(), &err);
    CHECK_CL(err);

    cl_mem d_discount = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * (size_t)N, discount.data(), &err);
    CHECK_CL(err);

    cl_mem d_shipday = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * (size_t)N, ship_day.data(), &err);
    CHECK_CL(err);

    // Launch sizing
    const size_t local = 256;
    const size_t global = ((size_t(N) + local - 1) / local) * local;
    const size_t num_groups = global / local;

    // Partial sums buffer

    cl_mem d_partials = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * num_groups, nullptr, &err);
    CHECK_CL(err);

    std::cout << "Workgroups: " << num_groups << " (local size: " << local << ")\n\n";



    // Warmup + repetitions
    const int WARMUP = 0;
    const int REPS = 1;

    std::vector<uint64_t> partials(num_groups);

    // ---------- Output CSV ----------
    std::ofstream csv("q6_results.csv");
    csv << "target_selectivity,window_days,start_day,end_day,achieved_selectivity,"
        << "kernel_ms_min,kernel_ms_median, kernel_ms_max,"
		<< "total_execution_time_median,overhead_ms,overhead_percentage, "
        << "gpu_to_cpu_transfer_in_ms,cpu_reduction_time_in_ms,"
        << "thread_utilization_percentage,wasted_threads_percentage,"
        << "data_efficiency_percentage,useful_data_MB, total_data_MB,"
        << "bandwidth_GB_per_sec,"
        << "cpu_result,gpu_result,abs_error\n";
    csv << std::fixed << std::setprecision(9);

    // Detailed timing
    struct TimingResult {
        double kernel_ms;
        double wall_ms;
        double d2h_ms;
        double cpu_finalize_ms;
    };

    auto launch_once_detailed = [&](int lo_day, int hi_day, std::vector<uint64_t>& out_partials) -> TimingResult {
        TimingResult timing;
        //Wall clock start
        auto wall_t0 = std::chrono::high_resolution_clock::now();
        CHECK_CL(clSetKernelArg(k, 0, sizeof(cl_mem), &d_quantity));
        CHECK_CL(clSetKernelArg(k, 1, sizeof(cl_mem), &d_price));
        CHECK_CL(clSetKernelArg(k, 2, sizeof(cl_mem), &d_discount));
        CHECK_CL(clSetKernelArg(k, 3, sizeof(cl_mem), &d_shipday));
        CHECK_CL(clSetKernelArg(k, 4, sizeof(cl_mem), &d_partials));
        CHECK_CL(clSetKernelArg(k, 5, sizeof(int), &lo_day));
        CHECK_CL(clSetKernelArg(k, 6, sizeof(int), &hi_day));
        CHECK_CL(clSetKernelArg(k, 7, sizeof(cl_uint), &N));

        cl_event evt;
        CHECK_CL(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, &evt));
        CHECK_CL(clFinish(q));

        cl_ulong t0 = 0, t1 = 0;
        CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
        CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr));
        timing.kernel_ms = double(t1 - t0) * 1e-6; // convert ns to ms
        clReleaseEvent(evt);

        // D2H timing

        auto d2h_t0 = std::chrono::high_resolution_clock::now();
        CHECK_CL(clEnqueueReadBuffer(q, d_partials, CL_TRUE, 0, sizeof(uint64_t) * num_groups, out_partials.data(), 0, nullptr, nullptr));
        auto d2h_t1 = std::chrono::high_resolution_clock::now();
        timing.d2h_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(d2h_t1 - d2h_t0).count();

        // CPU finalize timing (summing partials)
        auto cpu_finalize_t0 = std::chrono::high_resolution_clock::now();
        uint64_t gpu_sum_cents = 0;
        for (size_t i = 0; i < (size_t)num_groups; i++) {
            gpu_sum_cents += out_partials[i];
        }
        (void)gpu_sum_cents;
        auto cpu_finalize_t1 = std::chrono::high_resolution_clock::now();
        timing.cpu_finalize_ms = std::chrono::duration<double, std::milli>(cpu_finalize_t1 - cpu_finalize_t0).count();

        //wall clock end
        auto wall_t1 = std::chrono::high_resolution_clock::now();
        timing.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
        return timing;
        };

    // ---------- Sweep targets ----------
    for (double s : targets) {
        int W = calibrate_W_for_target(quantity_sorted, ship_day_sorted, discount_sorted, D0_day, s);

        if (W < 0) {
            // D0 beyond max; achieved is 0
            csv << s << ",NA," << D0_day << ",NA,0,NA,NA,NA,NA,NA\n";
            std::cout << "target_s=" << s << " -> D0 beyond max shipdate, achieved_s=0\n";
            continue;
        }

        const int lo_day = D0_day;
        const int hi_day = D0_day + W; 

        // warmup
        for (int i = 0; i < WARMUP; i++) {
            TimingResult dummy = launch_once_detailed(lo_day, hi_day, partials);
            (void)dummy;
        }
        // measured reps
        std::vector<TimingResult> timings;
        timings.reserve(REPS);
        for (int r = 0; r < REPS; r++) {
            timings.push_back(launch_once_detailed(lo_day, hi_day, partials));
            
            
        }
        //sorted by kernel time

        std::sort(timings.begin(), timings.end(), [](const TimingResult& a, const TimingResult& b) {
            return a.kernel_ms < b.kernel_ms;
            });
        
		const size_t remove_count = timings.size() * 35 / 100;  

        std::vector<TimingResult> filtered_kernel;
        for (size_t i = remove_count; i < timings.size() - remove_count; i++) {
            filtered_kernel.push_back(timings[i]);
		}   

        const double ms_min = filtered_kernel.front().kernel_ms;
        const double ms_max = filtered_kernel.back().kernel_ms;
        const double ms_med = filtered_kernel[filtered_kernel.size() / 2].kernel_ms;

		std::vector<TimingResult> timings_by_wall = timings;
        std::sort(timings_by_wall.begin(), timings_by_wall.end(), [](const TimingResult& a, const TimingResult& b) {
            return a.wall_ms < b.wall_ms;
			});
		std::vector<TimingResult> filtered_wall;
        for (size_t i = remove_count; i < timings_by_wall.size() - remove_count; i++) {
            filtered_wall.push_back(timings_by_wall[i]);
        }
        
        const double wall_ms_med = filtered_wall[filtered_wall.size() / 2].wall_ms;
        const double d2h_ms_med = filtered_wall[filtered_wall.size() / 2].d2h_ms;
        const double cpu_finalize_ms_med = filtered_wall[filtered_wall.size() / 2].cpu_finalize_ms;
        const double overhead_ms = wall_ms_med - ms_med;
        const double overhead_pct = (overhead_ms / wall_ms_med) * 100.0;

        // Read back partials

        uint64_t gpu_sum_cents = 0;
        for (size_t i = 0; i < (size_t)num_groups; i++) {
            gpu_sum_cents += partials[i];
        }

        double gpu_sum = double(gpu_sum_cents) / 100.0; // convert back to dollars

        uint64_t cnt = cpu_q6_count(quantity, ship_day, discount, lo_day, hi_day);
        const double achieved_s = double(cnt) / double(N);

        const double thread_utilization_pct = achieved_s * 100.0;
        const double wasted_threads_pct = (1.0 - achieved_s) * 100.0;

        //data efficiency matrics
        const double row_size_bytes = sizeof(float) * 3 + sizeof(int); // quantity price + discount + shipday
        const double useful_data_MB = double(cnt) * row_size_bytes / 1e6;
        const double total_data_MB = double(N) * row_size_bytes / 1e6;
        const double data_efficiency_pct = compute_data_efficiency(cnt, N) * 100.0;


        // ----- CPU reference for correctness check------
        const double cpu_sum = cpu_q6(quantity, price, discount, ship_day, lo_day, hi_day);
        const double abs_err = std::abs(cpu_sum - gpu_sum);

        // bandwidth estimate: kernel reads price + discount + shipday
        const double bytes_read = double(N) * (sizeof(float) * 3 + sizeof(int));
        const double bytes_written = double(num_groups) * sizeof(uint64_t);
        const double total_bytes = bytes_read + bytes_written;
        const double theoritical_gbps_med = total_bytes / (ms_med * 1e6);

        csv << s << "," << W << "," << lo_day << "," << hi_day << ","
            << achieved_s << ","
            << ms_min << "," << ms_med << "," << ms_max << ","
            << wall_ms_med << "," << overhead_ms << "," << overhead_pct << ","
            << d2h_ms_med << "," << cpu_finalize_ms_med << ","
            << thread_utilization_pct << "," << wasted_threads_pct << ","
            << data_efficiency_pct << "," << useful_data_MB << "," << total_data_MB << ","
            << theoritical_gbps_med << ","
            << cpu_sum << "," <<gpu_sum << "," << abs_err << "\n";

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "target_s=" << s
            << "  W=" << W
            << "  achieved_s=" << std::setprecision(6) << achieved_s
            << "  kernel=" << std::setprecision(3) << ms_min << "/" << ms_max
            << "  wall=" << wall_ms_med << "ms"
            << "  overhead=" << std::setprecision(1) << overhead_pct << "%"
            << "  thread_util=" << thread_utilization_pct << "%"
            << "  data_eff=" << data_efficiency_pct << "%"
            << "  abs_err=" << std::setprecision(3) << abs_err
            << "\n";
    }

    csv.close();
    std::cout << "Wrote q6_results.csv\n";

    // ---------- Cleanup ----------
	clReleaseMemObject(d_quantity);
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





