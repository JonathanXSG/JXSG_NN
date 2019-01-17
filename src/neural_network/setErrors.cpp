#include "../../headers/NeuralNetwork.hpp"

void NeuralNetwork::setErrors() {
  switch(costFunctionType) {
    case(COST_MSE): this->setErrorMSE(); break;
    case(COST_CEE): this->setErrorMSE(); break;
    default: this->setErrorMSE(); break;
  }
}

//  Cost Function Mean Squared Error
void NeuralNetwork::setErrorMSE() {
  int outputLayerIndex                = this->layers.size() - 1;
  std::vector<double> outputNeurons = this->layers.at(outputLayerIndex)->getActivatedValues();

  this->error = 0.00;

  for(int i = 0; i < target.size(); i++) {
    double targetValue  = target.at(i);
    double oActivatedValue  = outputNeurons.at(i);

    // Loss function on each neuron
    // ( |targetValue - oActivatedValue| )^2 \ 2
//    TODO: Change the 0.5* to a variable depending on training
    errors.at(i)        = 0.5 * pow(fabs(targetValue - oActivatedValue), 2);
    derivedErrors.at(i) = (oActivatedValue - targetValue);

    this->error += errors.at(i);
  }
}

// Cost Function: Cross Entropy Error
void NeuralNetwork::setErrorCEE(){
  unsigned outputLayerIndex           = this->layers.size() - 1;
  std::vector<double> outputNeurons   = this->layers.at(outputLayerIndex)->getActivatedValues();

  this->error = 0.00;

  for(int i = 0; i < target.size(); i++) {
    double targetValue      = target.at(i);
    double oActivatedValue  = outputNeurons.at(i);

    // Loss function on each neuron
    errors.at(i)        = - targetValue*log(oActivatedValue);
    // Derivative of Loss function in relation to the output
    derivedErrors.at(i) = - targetValue / oActivatedValue;;

    this->error += errors.at(i);
  }
}
