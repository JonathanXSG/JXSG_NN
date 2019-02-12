#include "../../headers/NeuralNetwork.hpp"

void NeuralNetwork::setErrors() {
    switch (costFunctionType) {
        case (MSE):
            this->setErrorMSE();
            break;
        case (CEE):
            this->setErrorCEE();
            break;
        default:
            this->setErrorMSE();
            break;
    }
}

//  Cost Function Mean Squared Error
void NeuralNetwork::setErrorMSE() {
    unsigned outputLayerIndex = this->layers.size() - 1;
    std::vector<double> *outputNeurons = this->layers.at(outputLayerIndex)->getActivatedValues();

    this->error = 0.00;

    for (unsigned i = 0; i < target->size(); i++) {
        // Loss function on each neuron
        errors.at(i) = (outputNeurons->at(i) - target->at(i)) * (outputNeurons->at(i) - target->at(i));
        derivedErrors.at(i) = (outputNeurons->at(i) - target->at(i));
        this->error += errors.at(i);
    }
    this->error *= 0.5;
}

// Cost Function: Cross Entropy Error
void NeuralNetwork::setErrorCEE() {
    unsigned outputLayerIndex = this->layers.size() - 1;
    std::vector<double> *outputNeurons = this->layers.at(outputLayerIndex)->getActivatedValues();

    this->error = 0.00;

    for (unsigned i = 0; i < target->size(); i++) {
        // Loss function on each neuron
        errors.at(i) = -target->at(i) * log(outputNeurons->at(i));
        // Derivative of Loss function in relation to the output
        derivedErrors.at(i) = -target->at(i) / outputNeurons->at(i);;

        this->error += errors.at(i);
    }
}
