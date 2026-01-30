// microbench_opencl.cpp

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static void die(const std::string& msg) {
    std::cerr << "ERROR: " << msg << "\n";
    std::exit(1);
}

static void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed with OpenCL error " << err;
        die(oss.str());
    }
}

static double event_ms(cl_event evt) {
    cl_ulong start = 0, end = 0;
    check(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo(START)");
    check(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo(END)");
    // ns -> ms
    return double(end - start) * 1e-6;
}

static const char* KERNEL_SRC = R"CLC(
__kernel void scan_filter_count(__global const float* r,
                                const uint N,
                                const float thresh,
                                __global uint* out_count)
{
    uint gid = get_global_id(0);
    if (gid >= N) return;
    if (r[gid] < thresh) {
        atomic_inc(out_count);
    }
}

// Projection + expression + compaction:
__kernel void projection_expr_compact(__global const float* r,
                                      __global const float* price,
                                      __global const float* disc,
                                      const uint N,
                                      const float thresh,
                                      __global uint* out_count,
                                      __global float* out_y)
{
    uint gid = get_global_id(0);
    if (gid >= N) return;

    if (r[gid] < thresh) {
        uint idx = atomic_inc(out_count);
        float y = price[gid] * (1.0f - disc[gid]);
        out_y[idx] = y;
    }
}

// Pass 1 of scalar aggregation (SUM of fixed-point cents):
__kernel void scalar_agg_partials(__global const float* r,
                                  __global const float* price,
                                  __global const float* disc,
                                  const uint N,
                                  const float thresh,
                                  __global ulong* out_partials)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group = get_group_id(0);
    uint lsize = get_local_size(0);

    __local ulong lsum[256]; // requires local size <= 256
    ulong x = 0;

    if (gid < N && r[gid] < thresh) {
        float y = price[gid] * (1.0f - disc[gid]);
        // fixed-point cents
        ulong cents = (ulong)(y * 100.0f);
        x = cents;
    }

    lsum[lid] = x;
    barrier(CLK_LOCAL_MEM_FENCE);

    // reduce in local memory
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


__kernel void groupby_sum_count(__global const float* r,
                               __global const uint* keys,
                               __global const float* price,
                               __global const float* disc,
                               const uint N,
                               const uint G,
                               const float thresh,
                               __global uint* out_sum_cents,   
                               __global uint* out_count)       
{
    uint gid = get_global_id(0);
    if (gid >= N) return;

    if (r[gid] < thresh) {
        uint k = keys[gid];
        if (k < G) {
            float y = price[gid] * (1.0f - disc[gid]);
            uint cents = (uint)(y * 100.0f);
            atomic_add(&out_sum_cents[k], cents);
            atomic_inc(&out_count[k]);
        }
    }
}
)CLC";

struct Args {
    uint32_t N = 10'000'000;
    uint32_t runs = 10;
    uint32_t warmup = 1;
    uint32_t G = 8;
    std::string csv = "results.csv";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) die(std::string("Missing value for ") + name);
            return std::string(argv[++i]);
            };
        if (k == "--N") a.N = (uint32_t)std::stoul(need("--N"));
        else if (k == "--runs") a.runs = (uint32_t)std::stoul(need("--runs"));
        else if (k == "--warmup") a.warmup = (uint32_t)std::stoul(need("--warmup"));
        else if (k == "--G") a.G = (uint32_t)std::stoul(need("--G"));
        else if (k == "--csv") a.csv = need("--csv");
        else die("Unknown argument: " + k);
    }
    return a;
}

static cl_device_id pick_gpu_device() {
    cl_uint numPlatforms = 0;
    check(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs(count)");
    if (numPlatforms == 0) die("No OpenCL platforms found.");

    std::vector<cl_platform_id> platforms(numPlatforms);
    check(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");

    // Prefer a GPU device, ideally Intel.  heuristic: first GPU found.
    for (auto p : platforms) {
        cl_uint numDev = 0;
        cl_int err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDev);
        if (err != CL_SUCCESS || numDev == 0) continue;

        std::vector<cl_device_id> devs(numDev);
        check(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, numDev, devs.data(), nullptr), "clGetDeviceIDs(GPU)");
        return devs[0];
    }

    die("No GPU OpenCL device found (CL_DEVICE_TYPE_GPU).");
    return nullptr;
}

static std::string device_name(cl_device_id dev) {
    size_t sz = 0;
    check(clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &sz), "clGetDeviceInfo(NAME,size)");
    std::string s(sz, '\0');
    check(clGetDeviceInfo(dev, CL_DEVICE_NAME, sz, s.data(), nullptr), "clGetDeviceInfo(NAME)");
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

static void write_csv_header_if_needed(const std::string& path) {
    std::ifstream in(path);
    if (in.good() && in.peek() != std::ifstream::traits_type::eof()) return;

    std::ofstream out(path, std::ios::out);
    out << "operator,variant,N,selectivity_target,selectivity_achieved,groups_G,skew,run_id,"
        "kernel_ms,h2d_ms,d2h_ms,end2end_ms,rows_per_s,eff_GBps,correct\n";
}

static void append_csv(const std::string& path, const std::string& line) {
    std::ofstream out(path, std::ios::app);
    out << line << "\n";
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::cout << "N=" << args.N << " runs=" << args.runs << " warmup=" << args.warmup << " G=" << args.G << "\n";

    // Selectivity sweep 
    std::vector<float> selectivities = { 0.001f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 1.0f };

    // Generate synthetic data: r uniform, price uniform, disc uniform, keys uniform
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> U01(0.0f, 1.0f);
    std::uniform_real_distribution<float> Uprice(1.0f, 1000.0f);
    std::uniform_real_distribution<float> Udisc(0.0f, 0.1f);
    std::uniform_int_distribution<uint32_t> Ukey(0, args.G - 1);

    std::vector<float> h_r(args.N), h_price(args.N), h_disc(args.N);
    std::vector<uint32_t> h_keys(args.N);

    for (uint32_t i = 0; i < args.N; i++) {
        h_r[i] = U01(rng);
        h_price[i] = Uprice(rng);
        h_disc[i] = Udisc(rng);
        h_keys[i] = Ukey(rng);
    }

    // CPU references (for correctness)
    auto cpu_filter_count = [&](float s) -> uint64_t {
        uint64_t c = 0;
        for (uint32_t i = 0; i < args.N; i++) if (h_r[i] < s) c++;
        return c;
        };
    auto cpu_scalar_sum_cents = [&](float s) -> uint64_t {
        uint64_t sum = 0;
        for (uint32_t i = 0; i < args.N; i++) {
            if (h_r[i] < s) {
                float y = h_price[i] * (1.0f - h_disc[i]);
                sum += (uint64_t)(y * 100.0f);
            }
        }
        return sum;
        };
    auto cpu_groupby = [&](float s, uint32_t G, std::vector<uint64_t>& sum_cents, std::vector<uint64_t>& cnt) {
        sum_cents.assign(G, 0);
        cnt.assign(G, 0);
        for (uint32_t i = 0; i < args.N; i++) {
            if (h_r[i] < s) {
                uint32_t k = h_keys[i];
                float y = h_price[i] * (1.0f - h_disc[i]);
                sum_cents[k] += (uint64_t)(y * 100.0f);
                cnt[k] += 1;
            }
        }
        };

    // OpenCL setup
    cl_int err = CL_SUCCESS;
    cl_device_id dev = pick_gpu_device();
    std::cout << "Device: " << device_name(dev) << "\n";

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    check(err, "clCreateContext");

    // Command queue with profiling
#if CL_TARGET_OPENCL_VERSION >= 200
    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    check(err, "clCreateCommandQueue");

    // Program build
    const char* srcs[] = { KERNEL_SRC };
    size_t lens[] = { std::strlen(KERNEL_SRC) };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, lens, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSz);
        std::string log(logSz, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSz, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        check(err, "clBuildProgram");
    }

    cl_kernel k_filter = clCreateKernel(prog, "scan_filter_count", &err);
    check(err, "clCreateKernel(scan_filter_count)");
    cl_kernel k_proj = clCreateKernel(prog, "projection_expr_compact", &err);
    check(err, "clCreateKernel(projection_expr_compact)");
    cl_kernel k_partials = clCreateKernel(prog, "scalar_agg_partials", &err);
    check(err, "clCreateKernel(scalar_agg_partials)");
    cl_kernel k_groupby = clCreateKernel(prog, "groupby_sum_count", &err);
    check(err, "clCreateKernel(groupby_sum_count)");

    // Buffers
    cl_mem d_r = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(float) * args.N, nullptr, &err);
    check(err, "clCreateBuffer(d_r)");
    cl_mem d_price = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(float) * args.N, nullptr, &err);
    check(err, "clCreateBuffer(d_price)");
    cl_mem d_disc = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(float) * args.N, nullptr, &err);
    check(err, "clCreateBuffer(d_disc)");
    cl_mem d_keys = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(uint32_t) * args.N, nullptr, &err);
    check(err, "clCreateBuffer(d_keys)");

    // Output buffers 
    cl_mem d_count1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &err);
    check(err, "clCreateBuffer(d_count1)");
    cl_mem d_proj_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * args.N, nullptr, &err);
    check(err, "clCreateBuffer(d_proj_out)");

    // Partials for scalar reduce: one partial per work-group
    const size_t local = 256;
    size_t global = ((size_t)args.N + local - 1) / local * local;
    const size_t num_groups = global / local;

    cl_mem d_partials = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint64_t) * num_groups, nullptr, &err);
    check(err, "clCreateBuffer(d_partials)");

    // Group-by outputs
    cl_mem d_gsum = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t) * args.G, nullptr, &err);
    check(err, "clCreateBuffer(d_gsum)");
    cl_mem d_gcnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t) * args.G, nullptr, &err);
    check(err, "clCreateBuffer(d_gcnt)");

    // Upload inputs 
    cl_event ev_w1, ev_w2, ev_w3, ev_w4;
    err = clEnqueueWriteBuffer(q, d_r, CL_FALSE, 0, sizeof(float) * args.N, h_r.data(), 0, nullptr, &ev_w1);
    check(err, "clEnqueueWriteBuffer(d_r)");
    err = clEnqueueWriteBuffer(q, d_price, CL_FALSE, 0, sizeof(float) * args.N, h_price.data(), 0, nullptr, &ev_w2);
    check(err, "clEnqueueWriteBuffer(d_price)");
    err = clEnqueueWriteBuffer(q, d_disc, CL_FALSE, 0, sizeof(float) * args.N, h_disc.data(), 0, nullptr, &ev_w3);
    check(err, "clEnqueueWriteBuffer(d_disc)");
    err = clEnqueueWriteBuffer(q, d_keys, CL_FALSE, 0, sizeof(uint32_t) * args.N, h_keys.data(), 0, nullptr, &ev_w4);
    check(err, "clEnqueueWriteBuffer(d_keys)");
    clFinish(q);

    double h2d_ms_inputs = event_ms(ev_w1) + event_ms(ev_w2) + event_ms(ev_w3) + event_ms(ev_w4);
    clReleaseEvent(ev_w1); clReleaseEvent(ev_w2); clReleaseEvent(ev_w3); clReleaseEvent(ev_w4);

    std::cout << "Input H2D total (ms): " << h2d_ms_inputs << "\n";

    // CSV header
    write_csv_header_if_needed(args.csv);

    auto zero_u32 = [&](cl_mem buf, size_t bytes) {
        std::vector<uint8_t> zeros(bytes, 0);
        cl_event ev;
        check(clEnqueueWriteBuffer(q, buf, CL_FALSE, 0, bytes, zeros.data(), 0, nullptr, &ev), "zero buffer");
        clFinish(q);
        clReleaseEvent(ev);
        };

    // Helper: run filter_count
    auto run_filter = [&](float s, uint32_t run_id) {
        // reset count
        uint32_t zero = 0;
        cl_event ev_zero;
        check(clEnqueueWriteBuffer(q, d_count1, CL_FALSE, 0, sizeof(uint32_t), &zero, 0, nullptr, &ev_zero),
            "write zero count");
        clFinish(q);
        double h2d_ms = event_ms(ev_zero);
        clReleaseEvent(ev_zero);

        // set args
        check(clSetKernelArg(k_filter, 0, sizeof(cl_mem), &d_r), "setArg r");
        check(clSetKernelArg(k_filter, 1, sizeof(uint32_t), &args.N), "setArg N");
        check(clSetKernelArg(k_filter, 2, sizeof(float), &s), "setArg thresh");
        check(clSetKernelArg(k_filter, 3, sizeof(cl_mem), &d_count1), "setArg out_count");

        auto t0 = std::chrono::steady_clock::now();
        cl_event ev_k;
        check(clEnqueueNDRangeKernel(q, k_filter, 1, nullptr, &global, &local, 0, nullptr, &ev_k), "enqueue kernel");
        clFinish(q);
        double kernel_ms = event_ms(ev_k);
        clReleaseEvent(ev_k);

        // read back count
        uint32_t out = 0;
        cl_event ev_r;
        check(clEnqueueReadBuffer(q, d_count1, CL_FALSE, 0, sizeof(uint32_t), &out, 0, nullptr, &ev_r),
            "read count");
        clFinish(q);
        double d2h_ms = event_ms(ev_r);
        clReleaseEvent(ev_r);

        auto t1 = std::chrono::steady_clock::now();
        double end2end_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double sel_ach = double(out) / double(args.N);

        // correctness check
        uint64_t cpu = cpu_filter_count(s);
        bool correct = (uint64_t(out) == cpu);

        // read r + write count (tiny)
        double bytes = double(args.N) * sizeof(float);
        double rows_per_s = double(args.N) / (kernel_ms / 1000.0);
        double eff_GBps = bytes / (kernel_ms / 1000.0) / 1e9;

        std::ostringstream line;
        line << "scan_filter_count,base," << args.N << "," << s << "," << sel_ach << ",0,uniform," << run_id << ","
            << kernel_ms << "," << h2d_ms << "," << d2h_ms << "," << end2end_ms << ","
            << rows_per_s << "," << eff_GBps << "," << (correct ? 1 : 0);
        append_csv(args.csv, line.str());
        };

    auto run_projection = [&](float s, uint32_t run_id) {
        // reset count
        uint32_t zero = 0;
        cl_event ev_zero;
        check(clEnqueueWriteBuffer(q, d_count1, CL_FALSE, 0, sizeof(uint32_t), &zero, 0, nullptr, &ev_zero),
            "write zero count");
        clFinish(q);
        double h2d_ms = event_ms(ev_zero);
        clReleaseEvent(ev_zero);

        check(clSetKernelArg(k_proj, 0, sizeof(cl_mem), &d_r), "setArg r");
        check(clSetKernelArg(k_proj, 1, sizeof(cl_mem), &d_price), "setArg price");
        check(clSetKernelArg(k_proj, 2, sizeof(cl_mem), &d_disc), "setArg disc");
        check(clSetKernelArg(k_proj, 3, sizeof(uint32_t), &args.N), "setArg N");
        check(clSetKernelArg(k_proj, 4, sizeof(float), &s), "setArg thresh");
        check(clSetKernelArg(k_proj, 5, sizeof(cl_mem), &d_count1), "setArg out_count");
        check(clSetKernelArg(k_proj, 6, sizeof(cl_mem), &d_proj_out), "setArg out_y");

        auto t0 = std::chrono::steady_clock::now();
        cl_event ev_k;
        check(clEnqueueNDRangeKernel(q, k_proj, 1, nullptr, &global, &local, 0, nullptr, &ev_k), "enqueue kernel");
        clFinish(q);
        double kernel_ms = event_ms(ev_k);
        clReleaseEvent(ev_k);

        // read back count 
        uint32_t out_count = 0;
        cl_event ev_r;
        check(clEnqueueReadBuffer(q, d_count1, CL_FALSE, 0, sizeof(uint32_t), &out_count, 0, nullptr, &ev_r),
            "read count");
        clFinish(q);
        double d2h_ms = event_ms(ev_r);
        clReleaseEvent(ev_r);

        auto t1 = std::chrono::steady_clock::now();
        double end2end_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double sel_ach = double(out_count) / double(args.N);

        // compare GPU count to CPU count 
        uint64_t cpu = cpu_filter_count(s);
        bool correct = (uint64_t(out_count) == cpu);

        // read r+price+disc, write out_y for passing
        double bytes = double(args.N) * (sizeof(float) * 3) + double(out_count) * sizeof(float);
        double rows_per_s = double(args.N) / (kernel_ms / 1000.0);
        double eff_GBps = bytes / (kernel_ms / 1000.0) / 1e9;

        std::ostringstream line;
        line << "projection_expr_compact,base," << args.N << "," << s << "," << sel_ach << ",0,uniform," << run_id << ","
            << kernel_ms << "," << h2d_ms << "," << d2h_ms << "," << end2end_ms << ","
            << rows_per_s << "," << eff_GBps << "," << (correct ? 1 : 0);
        append_csv(args.csv, line.str());
        };

    auto run_scalar_agg = [&](float s, uint32_t run_id) {
        // kernel overwrites each group output
        check(clSetKernelArg(k_partials, 0, sizeof(cl_mem), &d_r), "setArg r");
        check(clSetKernelArg(k_partials, 1, sizeof(cl_mem), &d_price), "setArg price");
        check(clSetKernelArg(k_partials, 2, sizeof(cl_mem), &d_disc), "setArg disc");
        check(clSetKernelArg(k_partials, 3, sizeof(uint32_t), &args.N), "setArg N");
        check(clSetKernelArg(k_partials, 4, sizeof(float), &s), "setArg thresh");
        check(clSetKernelArg(k_partials, 5, sizeof(cl_mem), &d_partials), "setArg out_partials");

        auto t0 = std::chrono::steady_clock::now();
        cl_event ev_k;
        check(clEnqueueNDRangeKernel(q, k_partials, 1, nullptr, &global, &local, 0, nullptr, &ev_k), "enqueue kernel");
        clFinish(q);
        double kernel_ms = event_ms(ev_k);
        clReleaseEvent(ev_k);

        // read back partials and finalize on CPU
        std::vector<uint64_t> h_partials(num_groups, 0);
        cl_event ev_r;
        check(clEnqueueReadBuffer(q, d_partials, CL_FALSE, 0, sizeof(uint64_t) * num_groups, h_partials.data(), 0, nullptr, &ev_r),
            "read partials");
        clFinish(q);
        double d2h_ms = event_ms(ev_r);
        clReleaseEvent(ev_r);

        uint64_t gpu_sum = 0;
        for (auto v : h_partials) gpu_sum += v;

        auto t1 = std::chrono::steady_clock::now();
        double end2end_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        uint64_t cpu_sum = cpu_scalar_sum_cents(s);
        bool correct = (gpu_sum == cpu_sum);


        uint64_t cpu_cnt = cpu_filter_count(s);
        double sel_ach = double(cpu_cnt) / double(args.N);

        // read r+price+disc, write partials
        double bytes = double(args.N) * (sizeof(float) * 3) + double(num_groups) * sizeof(uint64_t);
        double rows_per_s = double(args.N) / (kernel_ms / 1000.0);
        double eff_GBps = bytes / (kernel_ms / 1000.0) / 1e9;

        std::ostringstream line;
        line << "scalar_agg_sum,fixedpoint_reduce," << args.N << "," << s << "," << sel_ach << ",0,uniform," << run_id << ","
            << kernel_ms << ",0," << d2h_ms << "," << end2end_ms << ","
            << rows_per_s << "," << eff_GBps << "," << (correct ? 1 : 0);
        append_csv(args.csv, line.str());
        };

    auto run_groupby = [&](float s, uint32_t run_id) {
        // zero outputs (G is small)
        zero_u32(d_gsum, sizeof(uint32_t) * args.G);
        zero_u32(d_gcnt, sizeof(uint32_t) * args.G);

        check(clSetKernelArg(k_groupby, 0, sizeof(cl_mem), &d_r), "setArg r");
        check(clSetKernelArg(k_groupby, 1, sizeof(cl_mem), &d_keys), "setArg keys");
        check(clSetKernelArg(k_groupby, 2, sizeof(cl_mem), &d_price), "setArg price");
        check(clSetKernelArg(k_groupby, 3, sizeof(cl_mem), &d_disc), "setArg disc");
        check(clSetKernelArg(k_groupby, 4, sizeof(uint32_t), &args.N), "setArg N");
        check(clSetKernelArg(k_groupby, 5, sizeof(uint32_t), &args.G), "setArg G");
        check(clSetKernelArg(k_groupby, 6, sizeof(float), &s), "setArg thresh");
        check(clSetKernelArg(k_groupby, 7, sizeof(cl_mem), &d_gsum), "setArg out_sum");
        check(clSetKernelArg(k_groupby, 8, sizeof(cl_mem), &d_gcnt), "setArg out_cnt");

        auto t0 = std::chrono::steady_clock::now();
        cl_event ev_k;
        check(clEnqueueNDRangeKernel(q, k_groupby, 1, nullptr, &global, &local, 0, nullptr, &ev_k), "enqueue kernel");
        clFinish(q);
        double kernel_ms = event_ms(ev_k);
        clReleaseEvent(ev_k);

        // read back results
        std::vector<uint32_t> h_sum(args.G, 0), h_cnt(args.G, 0);
        cl_event ev_r1, ev_r2;
        check(clEnqueueReadBuffer(q, d_gsum, CL_FALSE, 0, sizeof(uint32_t) * args.G, h_sum.data(), 0, nullptr, &ev_r1),
            "read gsum");
        check(clEnqueueReadBuffer(q, d_gcnt, CL_FALSE, 0, sizeof(uint32_t) * args.G, h_cnt.data(), 0, nullptr, &ev_r2),
            "read gcnt");
        clFinish(q);
        double d2h_ms = event_ms(ev_r1) + event_ms(ev_r2);
        clReleaseEvent(ev_r1); clReleaseEvent(ev_r2);

        auto t1 = std::chrono::steady_clock::now();
        double end2end_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // correctness vs CPU (fixed-point cents)
        std::vector<uint64_t> cpu_sum, cpu_cnt;
        cpu_groupby(s, args.G, cpu_sum, cpu_cnt);

        bool correct = true;
        uint64_t total_cnt = 0;
        for (uint32_t g = 0; g < args.G; g++) {
            total_cnt += h_cnt[g];

            if (uint64_t(h_sum[g]) != cpu_sum[g] || uint64_t(h_cnt[g]) != cpu_cnt[g]) {
                correct = false;
                break;
            }
        }
        double sel_ach = double(total_cnt) / double(args.N);

        // read r+keys+price+disc, write G arrays
        double bytes = double(args.N) * (sizeof(float) * 3 + sizeof(uint32_t)) + double(args.G) * sizeof(uint32_t) * 2;
        double rows_per_s = double(args.N) / (kernel_ms / 1000.0);
        double eff_GBps = bytes / (kernel_ms / 1000.0) / 1e9;

        std::ostringstream line;
        line << "groupby_sum_count,fixedpoint_atomics," << args.N << "," << s << "," << sel_ach << ","
            << args.G << ",uniform," << run_id << ","
            << kernel_ms << ",0," << d2h_ms << "," << end2end_ms << ","
            << rows_per_s << "," << eff_GBps << "," << (correct ? 1 : 0);
        append_csv(args.csv, line.str());
        };

    // Run suite: warmups + measured runs
    for (float s : selectivities) {
        std::cout << "\nSelectivity target: " << s << "\n";

        for (uint32_t w = 0; w < args.warmup; w++) {
            run_filter(s, 100000 + w);
            run_projection(s, 100000 + w);
            run_scalar_agg(s, 100000 + w);
            run_groupby(s, 100000 + w);
        }
        for (uint32_t r = 0; r < args.runs; r++) {
            run_filter(s, r);
            run_projection(s, r);
            run_scalar_agg(s, r);
            run_groupby(s, r);
        }
    }

    // Cleanup
    clReleaseMemObject(d_r);
    clReleaseMemObject(d_price);
    clReleaseMemObject(d_disc);
    clReleaseMemObject(d_keys);
    clReleaseMemObject(d_count1);
    clReleaseMemObject(d_proj_out);
    clReleaseMemObject(d_partials);
    clReleaseMemObject(d_gsum);
    clReleaseMemObject(d_gcnt);

    clReleaseKernel(k_filter);
    clReleaseKernel(k_proj);
    clReleaseKernel(k_partials);
    clReleaseKernel(k_groupby);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    std::cout << "\nDone. Results appended to: " << args.csv << "\n";
    return 0;
}
