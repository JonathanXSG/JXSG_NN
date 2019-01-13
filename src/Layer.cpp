#include "../headers/Layer.hpp"

Layer::Layer(int size, LAYER_TYPE layerType, NN_ACTIVATION activationType) {
  this->size = size;
  this->layerType = layerType;

  double totalActivated = 0.0;

  for(int i = 0; i < size; i++) {
    Neuron *n = new Neuron(0.000000000, activationType);
    if(layerType == OUTPUT) totalActivated += n->getActivatedValue();
    this->neurons.push_back(n);
  }

  // If the layer is the output and the activation function is SOFTMAX then 
  // some more computation is needed
  if(layerType == OUTPUT && activationType == A_SOFTMAX){
    for(Neuron *n : this->neurons){
      n->setActivatedVal(n->getActivatedValue() / totalActivated);
    }
  }
}

void Layer::setVal(int i, double v) {
  this->neurons.at(i)->setVal(v);
}

Matrix *Layer::matrixifyValues() {
  Matrix *m = new Matrix(1, this->neurons.size(), false);
  for(int i = 0; i < this->neurons.size(); i++) {
    m->setValue(0, i, this->neurons.at(i)->getVal());
  }   
  return m;
}

Matrix *Layer::matrixifyActivatedValues() {
  Matrix *m = new Matrix(1, this->neurons.size(), false);
    for(int i = 0; i < this->neurons.size(); i++) {
      m->setValue(0, i, this->neurons.at(i)->getActivatedValue());
    }   
  return m;
}

Matrix *Layer::matrixifyDerivedValues() {
  Matrix *m = new Matrix(1, this->neurons.size(), false);
  for(int i = 0; i < this->neurons.size(); i++) {
    m->setValue(0, i, this->neurons.at(i)->getDerivedValue());
  }
  return m;
}

std::vector<double> Layer::getActivatedValues() {
  std::vector<double> ret;

  for(int i = 0; i < this->neurons.size(); i++) {
    double v = this->neurons.at(i)->getActivatedValue();
    ret.push_back(v);
  }

  return ret;
}
