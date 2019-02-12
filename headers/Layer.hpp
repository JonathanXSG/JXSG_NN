#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include <algorithm>
#include "Matrix.hpp"

enum LayerType {
    Input,
    HiddenFullyConnected,
    HiddenConvolutional,
    HiddenBatchNorm,
    Output
};

enum ActivationFunc {
    TANH,
    SIGM,
    RELU,
    LeakyRELU,
    LINE,
    SOFTMAX
};

class Layer {
public:
    Layer(unsigned size, LayerType layerType, ActivationFunc activationType = RELU);

    void activate();

    void derive();

    inline void setVal(unsigned i, double v) {
        this->neurons->at(i) = v;
    }

    void setNeurons(std::vector<double>* neurons) {
        this->neurons = neurons;
    }

    Matrix *matrixifyValues() const;

    Matrix *matrixifyActivatedValues() const;

    Matrix *matrixifyDerivedValues() const;

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
    unsigned size;
    LayerType layerType;
    ActivationFunc activationType;
    std::vector<double> *neurons;
    std::vector<double> *activatedNeurons;
    std::vector<double> *derivedNeurons;
};

#endif
