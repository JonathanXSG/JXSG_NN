#include "../headers/Layer.hpp"

Layer::Layer(int size, LAYER_TYPE layerType, NN_ACTIVATION activationType) {
    this->size = size;
    this->layerType = layerType;
    this->activationType = activationType;

    this->neurons = new std::vector<double>(size , 0.000000000);
    this->activatedNeurons = new std::vector<double>(size , 0.000000000);
    this->derivedNeurons = new std::vector<double>(size , 0.000000000);
}

void Layer::activate() {
    switch (activationType) {
        case A_TANH:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = tanh(this->neurons->at(i));
            }
            break;
        }
        case A_SIGM:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = (1.0 / (1.0 + exp(-this->neurons->at(i))));
            }
            break;
        }
        case A_RELU:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = (this->neurons->at(i) > 0.0 ? this->neurons->at(i) : 0.0);
            }
            break;
        }
        case A_LeakyRELU:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = this->neurons->at(i) > 0.0 ? this->neurons->at(i) : this->neurons->at(i) / 100;
            }
            break;
        }
        case A_LINE:{
            for (unsigned i = 0; i < this->neurons->size(); i++) {
                this->activatedNeurons->at(i) = this->neurons->at(i);
            }
            break;
        }
        case A_SOFTMAX: {
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
        case A_TANH:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = (1.0 - (this->activatedNeurons->at(i) * this->activatedNeurons->at(i)));
            }
            break;
        case A_RELU:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = this->activatedNeurons->at(i) > 0 ? 1.0 : 0.0;
            }
            break;
        case A_LeakyRELU:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = this->activatedNeurons->at(i) > 0.0 ? 1.0 : 1.0/100.0;
            }
            break;
        case A_SIGM:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = (this->activatedNeurons->at(i) * (1.0 - this->activatedNeurons->at(i)));
            }
            break;
        case A_LINE:
            for(unsigned i=0; i<this->neurons->size(); i++){
                this->derivedNeurons->at(i) = 1.0;
            }
            break;
        case A_SOFTMAX:
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

Matrix *Layer::matrixifyValues() {
    auto *m = new Matrix(1, this->neurons->size(), false);
    for (unsigned i = 0; i < this->neurons->size(); i++) {
        m->setValue(0, i, this->neurons->at(i));
    }
    return m;
}

Matrix *Layer::matrixifyActivatedValues() {
    auto *m = new Matrix(1, this->neurons->size(), false);
    for (unsigned i = 0; i < this->neurons->size(); i++) {
        m->setValue(0, i, this->activatedNeurons->at(i));
    }
    return m;
}

Matrix *Layer::matrixifyDerivedValues() {
    auto *m = new Matrix(1, this->neurons->size(), false);
    for (unsigned i = 0; i < this->neurons->size(); i++) {
        m->setValue(0, i, this->derivedNeurons->at(i));
    }
    return m;
}

double generateRandomNumber() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-.0001, .0001);

    return (double) rand()/(double)RAND_MAX;
}
