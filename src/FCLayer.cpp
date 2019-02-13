#include "../headers/Layer.hpp"
#include "../headers/FCLayer.hpp"

FCLayer::FCLayer(unsigned index, unsigned size, LayerType layerType, ActivationFunc activationType, double bias){
    this->layerIndex = index;
    this->size = size;
    this->layerType = layerType;
    this->activationType = activationType;
    this->bias = bias;

    this->neurons = new Matrix(false, size);
    this->activatedNeurons = new Matrix(false, size);
    this->derivedNeurons = new Matrix(false, size);
}

void FCLayer::feedForward(const Layer *prevLayer, const Matrix *weights) {

//    Fills the neurons in this layer with the bias, if there is none, it will be 0/
    neurons->fillMatrix(bias);

    for (unsigned r = 0; r < weights->getHeight(); r++) {
        for (unsigned c = 0; c < weights->getWidth(); c++) {
            neurons->at(c) += prevLayer->getNeuronsActivated()->at(r) * weights->at(r, c);
        }
    }

    activate();
    derive();
}

void backPropagation(const Layer *prevLayer, const Matrix *weights);

