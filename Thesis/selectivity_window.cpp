// selectivity_window.cpp
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>


// Convert YYYY-MM-DD to {y,m,d}
struct YMD { int y, m, d; };

static YMD parse_ymd(const std::string& s) {
    // "1996-03-13"
    return { std::stoi(s.substr(0,4)), std::stoi(s.substr(5,2)), std::stoi(s.substr(8,2)) };
}

static bool is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

// Convert a date to a day index 
static int days_since_epoch(YMD dt) {
    static const int mdays_norm[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };

    int days = 0;
    // years
    for (int y = 1970; y < dt.y; y++) days += is_leap(y) ? 366 : 365;

    // months
    for (int m = 1; m < dt.m; m++) {
        days += mdays_norm[m - 1];
        if (m == 2 && is_leap(dt.y)) days += 1; // Feb in leap year
    }

    // days
    days += (dt.d - 1);
    return days;
}

static int yyyymmdd_from_dayindex(int dayIndex) {
    // Convert back to YYYYMMDD for printing.
    static const int mdays_norm[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };

    int y = 1970;
    int di = dayIndex;
    while (true) {
        int yd = is_leap(y) ? 366 : 365;
        if (di >= yd) { di -= yd; y++; }
        else break;
    }
    int m = 1;
    while (true) {
        int md = mdays_norm[m - 1];
        if (m == 2 && is_leap(y)) md += 1;
        if (di >= md) { di -= md; m++; }
        else break;
    }
    int d = di + 1;
    return y * 10000 + m * 100 + d;
}

static int parse_yyyymmdd(const std::string& s) {
    // s = "1996-03-13"
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: selectivity_window.exe <path_to_lineitem.tbl>\n";
        return 1;
    }
    const std::string path = argv[1];
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open: " << path << "\n";
        return 1;
    }

    // Target selectivities
    std::vector<double> targets = { 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0 };

    // Q6 anchor date D0
    const std::string D0_str = "1994-01-01";
    const int D0_day = days_since_epoch(parse_ymd(D0_str));

    std::vector<int> ship_day;
    ship_day.reserve(6'500'000);

    std::string line;
    size_t N = 0;

    int minYMD = std::numeric_limits<int>::max();
    int maxYMD = std::numeric_limits<int>::min();

    while (std::getline(in, line)) {
   
        int field = 0;
        size_t start = 0;
        size_t pos = 0;
        std::string shipdate_str;

        while (true) {
            pos = line.find('|', start);
            if (pos == std::string::npos) break;
            if (field == 10) { shipdate_str = line.substr(start, pos - start); break; }
            field++;
            start = pos + 1;
        }
        if (shipdate_str.size() != 10) {
            std::cerr << "Bad shipdate at row " << N << ": '" << shipdate_str << "'\n";
            return 1;
        }

        int ymd = parse_yyyymmdd(shipdate_str);
        minYMD = std::min(minYMD, ymd);
        maxYMD = std::max(maxYMD, ymd);

        int di = days_since_epoch(parse_ymd(shipdate_str));
        ship_day.push_back(di);
        N++;
    }

    std::cout << "Rows (N): " << N << "\n";
    std::cout << "Min shipdate: " << yyyymmdd_to_string(minYMD) << "\n";
    std::cout << "Max shipdate: " << yyyymmdd_to_string(maxYMD) << "\n";
    std::cout << "Q6 anchor D0: " << D0_str << "\n\n";

    std::sort(ship_day.begin(), ship_day.end());

    // ---------- Q1: cutoff dates for l_shipdate <= D ----------
    std::cout << "Q1 cutoff dates\n";
    std::cout << "target_s,cutoff_D,achieved_s\n";
    for (double s : targets) {
        size_t k = (size_t)std::ceil(s * (double)N);
        if (k < 1) k = 1;
        if (k > N) k = N;
        int cutoff_day = ship_day[k - 1];
        int cutoff_ymd = yyyymmdd_from_dayindex(cutoff_day);
        double achieved = (double)k / (double)N;
        std::cout << s << "," << yyyymmdd_to_string(cutoff_ymd) << "," << achieved << "\n";
    }
    std::cout << "\n";

    // ---------- Q6: window length W days for D0 <= shipdate < D0+W ----------
    
    std::cout << "Q6 selection window :\n";
    std::cout << "target_s,D0,window_days_W,achieved_s\n";

    for (double s : targets) {
        size_t targetCount = (size_t)std::ceil(s * (double)N);
        if (targetCount < 1) targetCount = 1;
        if (targetCount > N) targetCount = N;

        // count in [D0, D0+W)
        auto it0 = std::lower_bound(ship_day.begin(), ship_day.end(), D0_day);
        size_t idx0 = (size_t)std::distance(ship_day.begin(), it0);
        if (idx0 >= N) {
            std::cout << s << "," << D0_str << ",NA,0\n";
            continue;
        }

        // Binary search W 
        int minDay = ship_day.front();
        int maxDay = ship_day.back();
        int maxW = (maxDay - D0_day) + 1;
        if (maxW < 1) maxW = 1;

        int lo = 1, hi = maxW, ansW = maxW;
        size_t ansCount = 0;

        while (lo <= hi) {
            int midW = lo + (hi - lo) / 2;
            int endDay = D0_day + midW;
            auto itEnd = std::lower_bound(ship_day.begin(), ship_day.end(), endDay);
            size_t idxEnd = (size_t)std::distance(ship_day.begin(), itEnd);
            size_t count = (idxEnd > idx0) ? (idxEnd - idx0) : 0;

            if (count >= targetCount) {
                ansW = midW;
                ansCount = count;
                hi = midW - 1;
            }
            else {
                lo = midW + 1;
            }
        }

        double achieved = (double)ansCount / (double)N;
        std::cout << s << "," << D0_str << "," << ansW << "," << achieved << "\n";
    }

    return 0;
}
