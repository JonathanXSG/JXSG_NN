#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <iostream>
#include <math.h>

enum NN_ACTIVATION {
  A_TANH        = 0,
  A_RELU        = 1,
  A_SIGM        = 2,
  A_LeakyRELU   = 3,
  A_LINE        = 4,
  A_SOFTMAX     = 5
};

class Neuron
{
public:
  Neuron(double value, NN_ACTIVATION activationType, double max = 0.0);

  void setVal(double value, double max = 0.0);
  void setActivatedVal(double value){
    this->activatedVal = value;
  };
  void activate(double max);
  void derive();

  // Getter
  double getVal() { return this->value; }
  double getActivatedValue() { return this->activatedVal; }
  double getDerivedValue() { return this->derivedVal; }

private:
  double value;

  double activatedVal;
  double derivedVal;

  NN_ACTIVATION activationType = A_RELU;
};

#endif
