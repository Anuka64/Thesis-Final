// Q6_ControlledSelectivity_WithCPUCheck.cpp
// TPC-H Q6 on OpenCL GPU with controlled selectivity
// Includes CPU correctness validation (single window, no performance comparison)

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>

// ---------------------------
// Parse YYYY-MM-DD -> YYYYMMDD
int parse_date(const std::string& date_str) {
    int y, m, d;
    char c1, c2;
    std::istringstream ss(date_str);
    ss >> y >> c1 >> m >> c2 >> d;
    return y * 10000 + m * 100 + d;
}

// ---------------------------
// Load TPC-H lineitem.tbl
void load_lineitem_csv(
    const std::string& filename,
    std::vector<float>& l_extendedprice,
    std::vector<float>& l_discount,
    std::vector<int>& l_quantity,
    std::vector<int>& l_shipdate)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << "\n";
        return;
    }

    std::string line, token;
    while (std::getline(file, line)) {
        std::istringstream ss(line);

        for (int i = 0; i < 4; i++) std::getline(ss, token, '|'); // skip keys

        std::getline(ss, token, '|'); int quantity = std::stoi(token);
        std::getline(ss, token, '|'); float price = std::stof(token);
        std::getline(ss, token, '|'); float discount = std::stof(token);

        for (int i = 0; i < 3; i++) std::getline(ss, token, '|'); // skip tax/status

        std::getline(ss, token, '|'); int shipdate = parse_date(token);

        l_quantity.push_back(quantity);
        l_extendedprice.push_back(price);
        l_discount.push_back(discount);
        l_shipdate.push_back(shipdate);
    }
}

// ---------------------------
// CPU reference Q6 (correctness only)
double q6_cpu_revenue(
    const std::vector<float>& price,
    const std::vector<float>& discount,
    const std::vector<int>& quantity,
    const std::vector<int>& shipdate,
    int shipdate_start,
    int shipdate_end)
{
    double revenue = 0.0;
    for (size_t i = 0; i < price.size(); i++) {
        if (shipdate[i] >= shipdate_start &&
            shipdate[i] < shipdate_end &&
            discount[i] >= 0.05f && discount[i] <= 0.07f &&
            quantity[i] < 24)
        {
            revenue += (double)price[i] * (double)discount[i];
        }
    }
    return revenue;
}

// ---------------------------
// OpenCL error helper
static void check_cl(cl_int err, const char* where) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error " << err << " at " << where << "\n";
        std::exit(1);
    }
}

// ---------------------------
// OpenCL kernel (Q6)
const char* kernelSource =
"__kernel void query6_gpu(__global const float* price,\n"
"                         __global const float* discount,\n"
"                         __global const int* quantity,\n"
"                         __global const int* shipdate,\n"
"                         const int shipdate_start,\n"
"                         const int shipdate_end,\n"
"                         __global float* out)\n"
"{\n"
"  int i = get_global_id(0);\n"
"  if (shipdate[i] >= shipdate_start && shipdate[i] < shipdate_end &&\n"
"      discount[i] >= 0.05f && discount[i] <= 0.07f &&\n"
"      quantity[i] < 24)\n"
"    out[i] = price[i] * discount[i];\n"
"  else\n"
"    out[i] = 0.0f;\n"
"}\n";

// ---------------------------
// Median helper
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------
// Main
int main() {
    std::vector<float> price, discount;
    std::vector<int> quantity, shipdate;

    load_lineitem_csv("C:\\TPC-H-V3.0.1\\dbgen\\lineitem.tbl",
        price, discount, quantity, shipdate);

    const size_t N = price.size();
    std::cout << "Loaded " << N << " rows\n";

    std::vector<std::pair<int, int>> windows = {
        {19940101, 19940108},
        {19940101, 19940201},
        {19940101, 19940401},
        {19940101, 19950101}   // full window (used for CPU check)
    };

    // ---------------- OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    check_cl(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
    check_cl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");

    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check_cl(err, "clCreateContext");

    cl_command_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    check_cl(err, "clCreateCommandQueue");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    check_cl(err, "clCreateProgram");

    check_cl(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram");
    cl_kernel kernel = clCreateKernel(program, "query6_gpu", &err);
    check_cl(err, "clCreateKernel");

    cl_mem d_price = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, price.data(), &err);
    cl_mem d_discount = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, discount.data(), &err);
    cl_mem d_quantity = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * N, quantity.data(), &err);
    cl_mem d_shipdate = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * N, shipdate.data(), &err);
    cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * N, nullptr, &err);

    check_cl(err, "clCreateBuffer");

    // Static args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_price);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_discount);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_quantity);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_shipdate);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_out);

    std::vector<float> out(N);

    std::cout << "\nshipdate_start,shipdate_end,selectivity,revenue,median_ms\n";

    for (const auto& w : windows) {
        int start = w.first, end = w.second;

        clSetKernelArg(kernel, 4, sizeof(int), &start);
        clSetKernelArg(kernel, 5, sizeof(int), &end);

        // warmup
        for (int i = 0; i < 2; i++)
            clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &N, nullptr, 0, nullptr, nullptr);

        std::vector<double> times;
        for (int r = 0; r < 15; r++) {
            cl_event evt;
            clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &N, nullptr, 0, nullptr, &evt);
            clWaitForEvents(1, &evt);

            cl_ulong t0, t1;
            clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
            clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
            times.push_back((t1 - t0) * 1e-6);
            clReleaseEvent(evt);
        }

        clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, sizeof(float) * N, out.data(), 0, nullptr, nullptr);

        double revenue = 0.0;
        size_t matched = 0;
        for (float v : out) {
            revenue += v;
            if (v != 0.0f) matched++;
        }

        double sel = (double)matched / (double)N;
        double med = median(times);

        std::cout << start << "," << end << "," << sel << "," << revenue << "," << med << "\n";

        // -------- CPU correctness check ONLY for full window
        if (end == 19950101) {
            double cpu_rev = q6_cpu_revenue(price, discount, quantity, shipdate, start, end);
            double diff = std::fabs(cpu_rev - revenue);
            double rel = diff / std::max(1.0, std::fabs(cpu_rev));

            std::cout << "\nCPU correctness check (full window):\n";
            std::cout << "CPU revenue = " << cpu_rev << "\n";
            std::cout << "GPU revenue = " << revenue << "\n";
            std::cout << "Relative error = " << rel << "\n";
            std::cout << (rel < 1e-4 ? "[PASS]\n" : "[FAIL]\n");
        }
    }

    // Cleanup
    clReleaseMemObject(d_price);
    clReleaseMemObject(d_discount);
    clReleaseMemObject(d_quantity);
    clReleaseMemObject(d_shipdate);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
