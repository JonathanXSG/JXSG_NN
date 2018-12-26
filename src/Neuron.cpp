#include "../headers/Neuron.hpp"

Neuron::Neuron(double val) {
  this->setVal(val); 
}

Neuron::Neuron(double val, int activationType) {
  this->activationType = activationType;
  this->setVal(val); 
}

void Neuron::setVal(double val) {
  this->val = val;
  activate();
  derive();
}

void Neuron::activate() {
  if(activationType == TANH) {
    this->activatedVal = tanh(this->val);
  } 
  else if(activationType == RELU) {
    this->activatedVal = this->val > 0 ? this->val : 0.0;
  } 
  else if(activationType == SIGM) {
    this->activatedVal = (1 / (1 + exp(-this->val)));
  } 
  else if(activationType == LINE) {
    this->activatedVal = this->val;
  } 
  else {
    this->activatedVal = (1 / (1 + exp(-this->val)));
  }
}

void Neuron::derive() {
  if(activationType == TANH) {
    this->derivedVal = (1.0 - (this->activatedVal * this->activatedVal));
  } 
  else if(activatedVal == RELU) {
    this->derivedVal = this->val > 0 ? 1.0 : 0.0;
  } 
  else if(activatedVal == SIGM) {
    this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
  } 
  else if(activationType == LINE) {
    this->activatedVal = 1.0;
  } 
  else {
    this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
  }
}

