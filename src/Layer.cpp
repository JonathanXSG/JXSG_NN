#include "../headers/Layer.hpp"

Layer::Layer(unsigned size, LayerType layerType, ActivationFunc activationType) {
    this->size = size;
    this->layerType = layerType;
    this->activationType = activationType;

    this->neurons = new std::vector<double>(size , 0.000000000);
    this->activatedNeurons = new std::vector<double>(size , 0.000000000);
    this->derivedNeurons = new std::vector<double>(size , 0.000000000);
}

void Layer::activate() {
    switch (activationType) {
        case TANH:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = tanh(this->neurons->at(i));
            }
            break;
        }
        case SIGM:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = (1.0 / (1.0 + exp(-this->neurons->at(i))));
            }
            break;
        }
        case RELU:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = (this->neurons->at(i) > 0.0 ? this->neurons->at(i) : 0.0);
            }
            break;
        }
        case LeakyRELU:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = this->neurons->at(i) > 0.0 ? this->neurons->at(i) : this->neurons->at(i) / 100;
            }
            break;
        }
        case LINE:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = this->neurons->at(i);
            }
            break;
        }
        case SOFTMAX: {
            double max = *max_element(this->neurons->begin(), this->neurons->end());
            double sum = std::accumulate(this->neurons->begin(), this->neurons->end(), 0.0);

            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = exp(this->neurons->at(i) - max) / sum;
            }
            break;
        }
        default:
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = (1.0 / (1.0 + exp(-this->neurons->at(i))));
            }
    }
}
void Layer::derive(){
    switch (activationType){
        case TANH:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = (1.0 - (this->activatedNeurons->at(i) * this->activatedNeurons->at(i)));
            }
            break;
        case SIGM:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = (this->activatedNeurons->at(i) * (1.0 - this->activatedNeurons->at(i)));
            }
            break;
        case RELU:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = this->activatedNeurons->at(i) > 0 ? 1.0 : 0.0;
            }
            break;
        case LeakyRELU:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = this->activatedNeurons->at(i) > 0.0 ? 1.0 : 1.0/100.0;
            }
            break;
        case LINE:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = 1.0;
            }
            break;
        case SOFTMAX:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = this->activatedNeurons->at(i) * (1.0 - this->activatedNeurons->at(i));
            }
            break;
        default:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = (this->activatedNeurons->at(i) * (1.0 - this->activatedNeurons->at(i)));
            }
    }
}

Matrix *Layer::matrixifyValues() const {
    auto *m = new Matrix(1, this->neurons->size(), false);
    for (unsigned i = 0; i < this->neurons->size(); i++) {
        m->at(0, i) = this->neurons->at(i);
    }
    return m;
}

Matrix *Layer::matrixifyActivatedValues() const {
    auto *m = new Matrix(1, this->neurons->size(), false);
    for (unsigned i = 0; i < this->neurons->size(); i++) {
        m->at(0, i) =  this->activatedNeurons->at(i);
    }
    return m;
}

Matrix *Layer::matrixifyDerivedValues() const {
    auto *m = new Matrix(1, this->neurons->size(), false);
    for (unsigned i = 0; i < this->neurons->size(); i++) {
        m->at(0, i) = this->derivedNeurons->at(i);
    }
    return m;
}
