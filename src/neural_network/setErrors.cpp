#include "../../headers/NeuralNetwork.hpp"

void NeuralNetwork::setErrors() {
  switch(costFunctionType) {
    case(COST_MSE): this->setErrorMSE(); break;
    default: this->setErrorMSE(); break;
  }
}

//  Cost Fuction Mean Squared Error
void NeuralNetwork::setErrorMSE() {
  int outputLayerIndex                = this->layers.size() - 1;
  std::vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNeurons();

  this->error = 0.00;

  for(int i = 0; i < target.size(); i++) {
    double targetValue  = target.at(i);
    double oActiatedvValue  = outputNeurons.at(i)->getActivatedVal();

    // ( |targetValue - oActivatedValue| )^2 \ 2
    errors.at(i)        = 0.5 * pow(abs((targetValue - oActiatedvValue)), 2);
    derivedErrors.at(i) = (oActiatedvValue - targetValue);

    this->error += errors.at(i);
  }
}
