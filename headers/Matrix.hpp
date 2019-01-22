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

        for (int i = 0; i < rows; i++) {
            std::vector<double> colValues;

            for (int j = 0; j < columns; j++) {
                double r = isRandom ? this->generateRandomNumber() : 0.00;
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

    Matrix *copy() {
        auto *m = new Matrix(this->rows, this->columns, false);

        for (unsigned i = 0; i < this->rows; i++) {
            for (unsigned j = 0; j < this->columns; j++) {
                m->setValue(i, j, this->getValue(i, j));
            }
        }

        return m;
    }

    void printToConsole() {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->columns; j++) {
                std::cout << std::setprecision(2) << this->values.at(i).at(j) << "\t";
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
    double generateRandomNumber() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-.0001, .0001);

        return (double)(rand() % 6) / 10.0;
    }

    unsigned rows;
    unsigned columns;

    std::vector<std::vector<double> > values;
};

#endif
