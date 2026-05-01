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
