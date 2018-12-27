#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include "Neuron.hpp"
#include "Matrix.hpp"
enum LAYER_TYPE{
  INPUT,
  HIDDEN_FULLYCONNECTED,
  OUTPUT
};

class Layer
{
public:
  Layer(int size, LAYER_TYPE layerType, NN_ACTIVATION activationType = A_RELU);

  void setVal(int i, double v);

  Matrix *matrixifyValues();
  Matrix *matrixifyActivatedValues();
  Matrix *matrixifyDerivedValues();

  std::vector<double> getActivatedValues();

  std::vector<Neuron *> getNeurons() { 
    return this->neurons; 
  }
  void setNeuron(std::vector<Neuron *> neurons) { 
    this->neurons = neurons; 
  }

private:
  int size;
  LAYER_TYPE layerType;
  std::vector<Neuron *> neurons;
};

#endif
