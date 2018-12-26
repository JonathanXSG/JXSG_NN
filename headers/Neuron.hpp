#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <iostream>
#include <math.h>

enum ACTIVATION_TYPE{
  TANH = 0,
  RELU = 1,
  SIGM = 2,
  LINE = 3
}; 

class Neuron
{
public:
  Neuron(double val);
  Neuron(double val, int activationType);

  void setVal(double v);
  void activate();
  void derive();

  // Getter
  double getVal() { return this->val; }
  double getActivatedVal() { return this->activatedVal; }
  double getDerivedVal() { return this->derivedVal; }

private:
  double val;

  double activatedVal;
  double derivedVal;

  int activationType = 2;
};

#endif
