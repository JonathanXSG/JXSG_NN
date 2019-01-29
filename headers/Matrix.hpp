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
        this->values.reserve(rows*columns);

        std::mt19937 gen(rd());
        double const distributionRangeHalfWidth = (2.4 / rows*columns);
        double const standardDeviation = distributionRangeHalfWidth * 2 / 6;
        std::normal_distribution<> normalDistribution(0.0,distributionRangeHalfWidth);

        for (int i = 0; i < rows*columns; i++) {
            this->values.emplace_back(isRandom ? normalDistribution(gen) : 0.00);
        }
    }

    Matrix *transpose() {
        auto *m = new Matrix(this->columns, this->rows, false);

        for (unsigned i = 0; i < this->rows; i++) {
            for (unsigned j = 0; j < this->columns; j++) {
                m->setValue(j, i, this->at(i, j));
            }
        }

        return m;
    }

    void printToConsole() {
        for (unsigned i = 0; i < this->rows; i++) {
            for (unsigned j = 0; j < this->columns; j++) {
                std::cout << std::setprecision(5) << this->at(i, j) << "\t";
            }
            std::cout << std::endl;
        }
    }

    void setValue(unsigned r, unsigned c, double v) {
        this->values.at(r*std::min(rows,columns) + c) = v;
    }

    double &at(unsigned r, unsigned c) {
        return this->values.at(r*this->columns + c);
    }

    std::vector<std::vector<double>> getValues() {
        auto vect = std::vector<std::vector<double>>(
                rows,
                std::vector<double>(columns));
        for (unsigned i = 0; i < this->rows; i++) {
            for (unsigned j = 0; j < this->columns; j++) {
                vect[i][j] = this->at(i, j);
            }
        }
        return vect;
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
    std::random_device rd;

    std::vector<double> values;
};

#endif
