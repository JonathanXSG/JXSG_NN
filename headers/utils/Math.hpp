#ifndef _MATH_HPP_
#define _MATH_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>
#include "../Matrix.hpp"


namespace utils {
    class Math {
    public:
        static void multiplyMatrix(Matrix *a, Matrix *b, Matrix *c) {
            for (int i = 0; i < a->getRows(); i++) {
                for (int j = 0; j < b->getColumns(); j++) {
                    for (int k = 0; k < b->getRows(); k++) {
                        double p = a->at(i, k) * b->at(k, j);
                        double newVal = c->at(i, j) + p;
                        c->at(i, j ) = newVal;
                    }
                }
            }
        }
    };
}

#endif
