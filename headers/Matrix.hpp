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
        double const var = 0.5;
        double const variance = 1.0/sqrt(rows * 1.0);
        double const variance2 = 2.0/(1.0*(rows +columns));
        double const varianceRELU = sqrt(2.0/(1.0*rows));
        std::normal_distribution<> normalDistribution(0.0,1.0);

        for (int i = 0; i < rows*columns; i++) {
            this->values.emplace_back(isRandom ? (normalDistribution(gen) * variance) : 0.00);
        }
    }

    Matrix *transpose() {
        auto *m = new Matrix(this->columns, this->rows, false);

        for (unsigned i = 0; i < this->rows; i++) {
            for (unsigned j = 0; j < this->columns; j++) {
                m->at(j, i) = this->at(i, j);
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

    inline double &at(const unsigned r, const unsigned c) {
        return this->values.at(c * rows + r);
    }

    template <typename T>
    std::vector<T> flatten(const std::vector<std::vector<T>>& v) {
        std::size_t total_size = 0;
        for (const auto& sub : v)
            total_size += sub.size(); // I wish there was a transform_accumulate
        std::vector<T> result;
        result.reserve(total_size);
        for (const auto& sub : v)
            result.insert(result.end(), sub.begin(), sub.end());
        return result;
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

    const unsigned getRows(){
        return this->rows;
    }

    const unsigned getColumns(){
        return this->columns;
    }

private:
    unsigned rows;
    unsigned columns;
    std::random_device rd;

    std::vector<double> values;
};

#endif
