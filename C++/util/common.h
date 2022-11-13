#pragma once
#include <sys/time.h>
#include <vector>
#include <iostream>

template <typename T>
std::ostream& operator<<(std::ostream& out, std::vector<T>& c) {
    out<<"[";
    for (int i = 0; i < c.size(); ++i) {
        if (i > 0) {
            out<< ",\t";
        }
        out<< c[i];
    }
    out<<"]\n";
}


inline int64_t gettimeofday_us() {
    timeval now;
    gettimeofday(&now, NULL);
    return now.tv_sec * 1000000L + now.tv_usec;
}
