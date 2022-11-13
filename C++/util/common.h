#pragma once
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


