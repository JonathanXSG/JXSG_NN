#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include <algorithm>
#include "Matrix.hpp"

enum LayerType {
    Input,
    HiddenFullyConnected,
    HiddenConvolutional,
    HiddenBatchNorm,
    Output
};

enum ActivationFunc {
    TANH,
    SIGM,
    RELU,
    LeakyRELU,
    LINE,
    SOFTMAX
};

class Layer {
public:

    void activate() {
        switch (activationType) {
            case TANH:{
                for (unsigned i = 0; i < this->neurons->getWidth(); i++) {
                    this->activatedNeurons->at(i) = tanh(this->neurons->at(i));
                }
                break;
            }
            case SIGM:{
                for (unsigned i = 0; i < this->neurons->getWidth(); i++) {
                    this->activatedNeurons->at(i) = (1.0 / (1.0 + exp(-this->neurons->at(i))));
                }
                break;
            }
            case RELU:{
                for (unsigned i = 0; i < this->neurons->getWidth(); i++) {
                    this->activatedNeurons->at(i) = (this->neurons->at(i) > 0.0 ? this->neurons->at(i) : 0.0);
                }
                break;
            }
            case LeakyRELU:{
                for (unsigned i = 0; i < this->neurons->getWidth(); i++) {
                    this->activatedNeurons->at(i) = this->neurons->at(i) > 0.0 ? this->neurons->at(i) : this->neurons->at(i) / 100;
                }
                break;
            }
            case LINE:{
                for (unsigned i = 0; i < this->neurons->getWidth(); i++) {
                    this->activatedNeurons->at(i) = this->neurons->at(i);
                }
                break;
            }
            case SOFTMAX: {
                double max = this->neurons->maxValue();
                double sum = this->neurons->accumulate();

                for (unsigned i = 0; i < this->neurons->getWidth(); i++) {
                    this->activatedNeurons->at(i) = exp(this->neurons->at(i) - max) / sum;
                }
                break;
            }
            default:
                for (unsigned i = 0; i < this->neurons->getWidth(); i++) {
                    this->activatedNeurons->at(i) = (1.0 / (1.0 + exp(-this->neurons->at(i))));
                }
        }
    }
    void derive(){
        switch (activationType){
            case TANH:
                for(unsigned i=0; i< this->neurons->getWidth(); i++){
                    this->derivedNeurons->at(i) = (1.0 - (this->activatedNeurons->at(i) * this->activatedNeurons->at(i)));
                }
                break;
            case SIGM:
                for(unsigned i=0; i< this->neurons->getWidth(); i++){
                    this->derivedNeurons->at(i) = (this->activatedNeurons->at(i) * (1.0 - this->activatedNeurons->at(i)));
                }
                break;
            case RELU:
                for(unsigned i=0; i< this->neurons->getWidth(); i++){
                    this->derivedNeurons->at(i) = this->activatedNeurons->at(i) > 0 ? 1.0 : 0.0;
                }
                break;
            case LeakyRELU:
                for(unsigned i=0; i< this->neurons->getWidth(); i++){
                    this->derivedNeurons->at(i) = this->activatedNeurons->at(i) > 0.0 ? 1.0 : 1.0/100.0;
                }
                break;
            case LINE:
                for(unsigned i=0; i< this->neurons->getWidth(); i++){
                    this->derivedNeurons->at(i) = 1.0;
                }
                break;
            case SOFTMAX:
                for(unsigned i=0; i< this->neurons->getWidth(); i++){
                    this->derivedNeurons->at(i) = this->activatedNeurons->at(i) * (1.0 - this->activatedNeurons->at(i));
                }
                break;
            default:
                for(unsigned i=0; i< this->neurons->getWidth(); i++){
                    this->derivedNeurons->at(i) = (this->activatedNeurons->at(i) * (1.0 - this->activatedNeurons->at(i)));
                }
        }
    }

    virtual void feedForward(const Layer *prevLayer, const Matrix *weights) = 0;

    virtual void backPropagation(const Layer *prevLayer, const Matrix *weights) = 0;

    virtual Matrix * getNeurons() const { return neurons;}

    virtual Matrix * getNeuronsActivated() const { return neurons;}

    virtual Matrix *getNeuronsDerived() { return neurons;}


protected:
    unsigned size;
    unsigned layerIndex;
    LayerType layerType;
    ActivationFunc activationType;
    double bias;
    Matrix *neurons;
    Matrix *activatedNeurons;
    Matrix *derivedNeurons;
};

#endif
