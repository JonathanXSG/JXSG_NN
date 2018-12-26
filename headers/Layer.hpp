#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include "Neuron.hpp"
#include "Matrix.hpp"

class Layer
{
public:
  Layer(int size);
  Layer(int size, int activationType);
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
  std::vector<Neuron *> neurons;
};

#endif
