#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

class Matrix {
public:
    Matrix(unsigned rows, unsigned columns, bool isRandom) {
        this->rows = rows;
        this->columns = columns;
        std::random_device rd;
        std::mt19937 gen(rd());
        double const distributionRangeHalfWidth = (2.4 / 784);
        double const standardDeviation = distributionRangeHalfWidth * 2 / 6;
        std::normal_distribution<> normalDistribution(-0.1, 0.1);
//        std::normal_distribution<> normalDistribution(0.0, standardDeviation );

        for (int i = 0; i < rows; i++) {
            std::vector<double> colValues;

            for (int j = 0; j < columns; j++) {
                double r = isRandom ? normalDistribution(gen) : 0.00;
                colValues.push_back(r);
            }

            this->values.push_back(colValues);
        }
    }

    Matrix *transpose() {
        auto *m = new Matrix(this->columns, this->rows, false);

        for (unsigned i = 0; i < this->rows; i++) {
            for (unsigned j = 0; j < this->columns; j++) {
                m->setValue(j, i, this->getValue(i, j));
            }
        }

        return m;
    }

    void printToConsole() {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->columns; j++) {
                std::cout << std::setprecision(4) << this->values.at(i).at(j) << "\t";
            }
            std::cout << std::endl;
        }
    }

    void setValue(unsigned r, unsigned c, double v) {
        this->values.at(r).at(c) = v;
    }

    double getValue(unsigned r, unsigned c) {
        return this->values.at(r).at(c);
    }

    std::vector<std::vector<double> > getValues() {
        return this->values;
    }

    unsigned getRows() {
        return this->rows;
    }

    unsigned getColumns() {
        return this->columns;
    }

private:
    unsigned rows;
    unsigned columns;

    std::vector<std::vector<double> > values;
};

#endif
