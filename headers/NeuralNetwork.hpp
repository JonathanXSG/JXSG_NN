#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "json.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"

using json = nlohmann::json;

enum CostFunc {
    MSE,
    CEE
};

enum GradientDescent {
    Stochastic,
    MiniBatch,
    Batch
};

struct NNConfig {
    std::vector<unsigned> topology;
    double bias;
    double learningRate;
    double momentum;
    int epochs;
    int iterations;
    ActivationFunc hActivation;
    ActivationFunc oActivation;
    CostFunc costFunction;
    GradientDescent gradientDescent;
    int batch;
    std::string dataFile;
    std::string labelsFile;
    std::string weightsFile;
    std::string reportFile;
};

class NeuralNetwork {
public:
    explicit NeuralNetwork(NNConfig config);

    void train(
            std::vector<std::vector<double>>& input,
            std::vector<std::vector<double>>& target
    );

    void test(
            std::vector<std::vector<double>>& input,
            std::vector<std::vector<double>>& target
    );

    void setCurrentInput(std::vector<double>& input);

    void setCurrentTarget(std::vector<double>& target);

    void feedForward();

    void backPropagation();

    void setErrors();

    inline std::vector<double>* getNeurons(unsigned index) {
        return this->layers.at(index)->getNeurons();
    }

    inline std::vector<double>* getActivatedNeurons(unsigned index) {
        return this->layers.at(index)->getActivatedValues();
    }

    inline std::vector<double>* getDerivedNeurons(unsigned index) {
        return this->layers.at(index)->getActivatedValues();
    }

    inline Matrix *getNeuronMatrix(unsigned index) {
        return this->layers.at(index)->matrixifyValues();
    }

    inline Matrix *getActivatedNeuronMatrix(unsigned index) {
        return this->layers.at(index)->matrixifyActivatedValues();
    }

    inline Matrix *getDerivedNeuronMatrix(unsigned index) {
        return this->layers.at(index)->matrixifyDerivedValues();
    }

    inline Matrix *getWeightMatrix(unsigned index) {
        return this->weightMatrices.at(index);
    };

    inline void setLayer(unsigned indexLayer, std::vector<double>& neurons) {
        this->layers.at(indexLayer)->setNeurons(&neurons);
    }

    void saveWeights(std::string file);

    void loadWeights(std::string file);

    unsigned topologySize;
    ActivationFunc hiddenActivationType = RELU;
    ActivationFunc outputActivationType = SIGM;
    CostFunc costFunctionType = MSE;
    GradientDescent gradientDescent = Stochastic;

    std::vector<unsigned> topology;
    std::vector<Layer *> layers;
    std::vector<Matrix *> weightMatrices;
    std::vector<Matrix *> deltaMatrices;

    std::vector<double>* target;
    std::vector<double> errors;
    std::vector<double> derivedErrors;

    double error = 0;
    double bias = 1;
    double momentum;
    double learningRate;

    NNConfig config;

private:
    void setErrorMSE();

    void setErrorCEE();
};

#endif
