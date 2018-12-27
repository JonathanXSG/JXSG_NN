#include "../headers/Neuron.hpp"

Neuron::Neuron(double value, NN_ACTIVATION activationType, double max) {
  this->activationType = activationType;
  this->setVal(value, max); 
}

void Neuron::setVal(double value, double max) {
  this->value = value;
  activate(max);
  derive();
}

void Neuron::activate(double max) {
  if(activationType == A_TANH) {
    this->activatedVal = tanh(this->value);
  } 
  else if(activationType == A_SIGM) {
    this->activatedVal = (1 / (1 + exp(-this->value)));
  } 
  else if(activationType == A_RELU) {
    this->activatedVal = this->value >= 0 ? this->value : 0.0;
  } 
  else if(activationType == A_LeakyRELU) {
    this->activatedVal = this->value >= 0 ? this->value : this->value/100;
  } 
  else if(activationType == A_LINE) {
    this->activatedVal = this->value;
  } 
  else if(activationType == A_SOFTMAX) {
    this->activatedVal = exp(this->value - max);
  } 
  else {
    this->activatedVal = (1 / (1 + exp(-this->value)));
  }
}

void Neuron::derive() {
  if(activationType == A_TANH) {
    this->derivedVal = (1.0 - (this->activatedVal * this->activatedVal));
  } 
  else if(activationType == A_RELU) {
    this->derivedVal = this->activatedVal >= 0 ? 1.0 : 0.0;
  } 
  else if(activationType == A_LeakyRELU) {

  } 
  else if(activationType == A_SIGM) {
    this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
  } 
  else if(activationType == A_LINE) {
    this->derivedVal = 1.0;
  } 
  else if(activationType == A_SOFTMAX) {
    this->derivedVal = this->activatedVal * (1- this->activatedVal);
  } 
  else {
    this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
  }
}

