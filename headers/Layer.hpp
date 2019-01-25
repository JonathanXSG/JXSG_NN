#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include <algorithm>
#include "Matrix.hpp"

enum LAYER_TYPE {
    INPUT,
    HIDDEN_FULLYCONNECTED,
    OUTPUT
};

enum NN_ACTIVATION {
    A_TANH        = 0,
    A_RELU        = 1,
    A_SIGM        = 2,
    A_LeakyRELU   = 3,
    A_LINE        = 4,
    A_SOFTMAX     = 5
};

class Layer {
public:
    Layer(int size, LAYER_TYPE layerType, NN_ACTIVATION activationType = A_RELU);

    void activate();

    void derive();

    void setVal(unsigned i, double v) {
        this->neurons->at(i) = v;
    }

    void setNeurons(std::vector<double> *neurons) {
        this->neurons = neurons;
    }

    Matrix *matrixifyValues();

    Matrix *matrixifyActivatedValues();

    Matrix *matrixifyDerivedValues();

    std::vector<double> *getNeurons() {
        return this->neurons;
    }

    std::vector<double> *getActivatedValues() {
        return activatedNeurons;
    }

    std::vector<double> *getDerivedValues() {
        return derivedNeurons;
    }

private:
    int size;
    LAYER_TYPE layerType;
    NN_ACTIVATION activationType;
    std::vector<double> *neurons;
    std::vector<double> *activatedNeurons;
    std::vector<double> *derivedNeurons;
};

#endif
