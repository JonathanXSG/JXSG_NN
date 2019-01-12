#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include "json.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"

using json = nlohmann::json;

enum NN_COST {
  COST_MSE,
  COST_CEE
};

enum GRADIENT_DESCENT{
  STOCHASTIC = 0,
  MINI_BATCH = 1,
  BATCH = 2
};

struct ANNConfig {
  std::vector<int> topology;
  double bias;
  double learningRate;
  double momentum;
  int epoch;
  NN_ACTIVATION hActivation;
  NN_ACTIVATION oActivation;
  NN_COST cost;
  GRADIENT_DESCENT gradientDescent;
  int batch;
  std::string trainingFile;
  std::string labelsFile;
  std::string weightsFile;
};

class NeuralNetwork
{
public:
  NeuralNetwork(ANNConfig config);

  void train(
        std::vector< std::vector<double>> input, 
        std::vector< std::vector<double>> target
      );

  void setCurrentInput(std::vector<double> input);
  void setCurrentTarget(std::vector<double> target) { this->target = target; };

  void feedForward();
  void backPropagation();
  void setErrors();
  void printToConsole();

  std::vector<double> getActivatedValues(int index) { 
    return this->layers.at(index)->getActivatedValues(); 
  }

  Matrix *getNeuronMatrix(int index) { 
    return this->layers.at(index)->matrixifyValues();
  }
  Matrix *getActivatedNeuronMatrix(int index) { 
    return this->layers.at(index)->matrixifyActivatedValues(); 
  }
  Matrix *getDerivedNeuronMatrix(int index) { 
    return this->layers.at(index)->matrixifyDerivedValues(); 
  }
  Matrix *getWeightMatrix(int index) { 
    return new Matrix(*this->weightMatrices.at(index)); 
    };

  void setNeuronValue(int indexLayer, int indexNeuron, double value) { 
    this->layers.at(indexLayer)->setVal(indexNeuron, value); 
  }

  void saveWeights(std::string file);
  void loadWeights(std::string file);

  int topologySize;
  NN_ACTIVATION hiddenActivationType  = A_RELU;
  NN_ACTIVATION outputActivationType  = A_SIGM;
  NN_COST costFunctionType            = COST_MSE;
  GRADIENT_DESCENT gradientDescent    = STOCHASTIC;

  std::vector<int> topology;
  std::vector<Layer *> layers;
  std::vector<Matrix *> weightMatrices;
  std::vector<Matrix *> gradientMatrices;

  std::vector<double> input;
  std::vector<double> target;
  std::vector<double> errors;
  std::vector<double> derivedErrors;

  double error              = 0;
  double bias               = 1;
  double momentum;
  double learningRate;

  ANNConfig config;

private:
  void setErrorMSE();
  void setErrorCEE();
};

#endif
