
#include "../headers/ConvLayer.hpp"

ConvLayer::ConvLayer(unsigned index, unsigned size, unsigned kernel, unsigned field, unsigned stride, unsigned padding,
                              LayerType layerType, ActivationFunc activationType, double bias){
    this->layerIndex = index;
    this->size = size;
    this->layerType = layerType;
    this->activationType = activationType;
    this->bias = bias;
    this->kernel = kernel;
    this->kernel = field;
    this->kernel = stride;
    this->kernel = padding;

    this->neurons = new Matrix(false, size, size);
    this->activatedNeurons = new Matrix(false, size, size);
    this->derivedNeurons = new Matrix(false, size, size);
}

void ConvLayer::feedForward(const Layer *prevLayer, const Matrix *weights){
    //    Fills the neurons in this layer with the bias, if there is none, it will be 0/
    neurons->fillMatrix(bias);

    if((prevLayer->getNeuronsActivated()->getHeight() - field + (2*padding))/(stride+1) != size){
        std::cout << "WRONG" << std::endl;
    }

    for (unsigned h = 0; h < weights->getHeight(); h++) {
        for (unsigned w = 0; w < weights->getWidth(); w++) {
            for (unsigned d = 0; d < weights->getWidth(); d++) {
                neurons->at(h, w, d) +=
            }
                neurons->at(c) += prevLayer->getNeuronsActivated()->at(h) * weights->at(h, c);
        }
    }

    activate();
    derive();
}

void ConvLayer::backPropagation(const Layer *prevLayer, const Matrix *weights){

}