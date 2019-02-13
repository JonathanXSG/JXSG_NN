#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

class Matrix {
public:
    Matrix(bool isRandom, unsigned width, unsigned height = 1) {
        this->height = height;
        this->width = width;
        this->values.reserve(width*height);

        std::mt19937 gen(rd());
        double const var = 0.5;
        double const variance = 1.0/sqrt(height * 1.0);
        double const variance2 = 2.0/(1.0*(height + width));
        double const varianceRELU = sqrt(2.0/(1.0*height));
        std::normal_distribution<> normalDistribution(0.0,1.0);

        for (int i = 0; i < height*width*depth; i++) {
            this->values.emplace_back(isRandom ? (normalDistribution(gen) * variance) : 0.00);
        }
    }

    Matrix *transpose() {
        auto *m = new Matrix(false, this->height, this->width);

        for (unsigned i = 0; i < this->width; i++) {
            for (unsigned j = 0; j < this->height; j++) {
                m->at(j, i) = this->at(i, j);
            }
        }

        return m;
    }

    void printToConsole() {
        for (unsigned i = 0; i < this->height; i++) {
            for (unsigned j = 0; j < this->width; j++) {
                std::cout << std::setprecision(5) << this->at(i, j) << "\t";
            }
            std::cout << std::endl;
        }
    }

    inline double &at(const unsigned w, const unsigned h = 0) {
        return this->values.at(w + (h*width));
    }

    inline const double at(const unsigned w, const unsigned h = 0) const{
        return this->values.at(w + (h*width));
    }

    double maxValue() {
        return *max_element(this->values.begin(), this->values.end());
    }

    double accumulate() {
        return std::accumulate(this->values.begin(), this->values.end(), 0.0);
    }

    double sum(const unsigned w, const unsigned h, const unsigned w2, const unsigned h2){
        double value = 0;
        for(unsigned deltaY = h; deltaY <= h2 ; deltaY++){
            value += std::accumulate(this->values.begin() + index(w, deltaY),
                                     this->values.begin() + index(w2, deltaY), 0);
        }
        return value;
    }

    unsigned index(const unsigned h, const unsigned w = 0) const{
        return (w + (h*width));
    }

    void fillMatrix(double value){
        fill(this->values.begin(),
             this->values.end(),
             value);
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
                height,
                std::vector<double>(width));
        for (unsigned i = 0; i < this->height; i++) {
            for (unsigned j = 0; j < this->width; j++) {
                vect[i][j] = this->at(i, j);
            }
        }
        return vect;
    }

    const unsigned getHeight() const {
        return this->height;
    }

    const unsigned getWidth() const {
        return this->width;
    }

private:
    unsigned height;
    unsigned width;
    unsigned depth;
    std::random_device rd;

    std::vector<double> values;
};

#endif
