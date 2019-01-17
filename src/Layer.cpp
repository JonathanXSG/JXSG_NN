#include "../headers/Layer.hpp"

Layer::Layer(int size, LAYER_TYPE layerType, NN_ACTIVATION activationType) {
    this->size = size;
    this->layerType = layerType;
    this->activationType = activationType;

    this->neurons.assign(size , 0.000000000);
    this->activatedNeurons.assign(size , 0.000000000);
    this->derivedNeurons.assign(size , 0.000000000);
}

void Layer::activate() {
    switch (activationType) {
        case A_TANH:{
            for (int i = 0; i < this->neurons.size(); i++) {
                this->activatedNeurons[i] = tanh(this->neurons[i]);
            }
            break;
        }
        case A_SIGM:{
            for (int i = 0; i < this->neurons.size(); i++) {
                this->activatedNeurons[i] = (1.0 / (1.0 + exp(-this->neurons[i])));
            }
            break;
        }
        case A_RELU:{
            for (int i = 0; i < this->neurons.size(); i++) {
                this->activatedNeurons[i] = (this->neurons[i] > 0.0 ? this->neurons[i] : 0.0);
            }
            break;
        }
        case A_LeakyRELU:{
            for (int i = 0; i < this->neurons.size(); i++) {
                this->activatedNeurons[i] = this->neurons[i] > 0.0 ? this->neurons[i] : this->neurons[i] / 100;
            }
            break;
        }
        case A_LINE:{
            for (int i = 0; i < this->neurons.size(); i++) {
                this->activatedNeurons[i] = this->neurons[i];
            }
            break;
        }
        case A_SOFTMAX: {
            double max = *max_element(this->neurons.begin(), this->neurons.end());
            double sum = std::accumulate(this->neurons.begin(), this->neurons.end(), 0.0);

            for (int i = 0; i < this->neurons.size(); i++) {
                this->activatedNeurons[i] = exp(this->neurons[i] - max) / sum;
            }
            break;
        }
        default:
            for (int i = 0; i < this->neurons.size(); i++) {
                this->activatedNeurons[i] = (1.0 / (1.0 + exp(-this->neurons[i])));
            }
    }
}
void Layer::derive(){
        switch (activationType){
            case A_TANH:
                for(int i=0; i<this->neurons.size(); i++){
                    this->derivedNeurons[i] = (1.0 - (this->activatedNeurons[i] * this->activatedNeurons[i]));
                }
            break;
            case A_RELU:
                for(int i=0; i<this->neurons.size(); i++){
                    this->derivedNeurons[i] = this->activatedNeurons[i] > 0 ? 1.0 : 0.0;
                }
            break;
            case A_LeakyRELU:
                for(int i=0; i<this->neurons.size(); i++){
                    this->derivedNeurons[i] = this->activatedNeurons[i] > 0.0 ? 1.0 : 1.0/100.0;
                }
            break;
            case A_SIGM:
                for(int i=0; i<this->neurons.size(); i++){
                    this->derivedNeurons[i] = (this->activatedNeurons[i] * (1.0 - this->activatedNeurons[i]));
                }
            break;
            case A_LINE:
                for(int i=0; i<this->neurons.size(); i++){
                    this->derivedNeurons[i] = 1.0;
                }
            break;
            case A_SOFTMAX:
                for(int i=0; i<this->neurons.size(); i++){
                    this->derivedNeurons[i] = this->activatedNeurons[i] * (1.0- this->activatedNeurons[i]);
                }
            break;
            default:
                for(int i=0; i<this->neurons.size(); i++){
                    this->derivedNeurons[i] = (this->activatedNeurons[i] * (1.0 - this->activatedNeurons[i]));
                }
        }
}

Matrix *Layer::matrixifyValues() {
    auto *m = new Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < this->neurons.size(); i++) {
        m->setValue(0, i, this->neurons.at(i));
    }
    return m;
}

Matrix *Layer::matrixifyActivatedValues() {
    auto *m = new Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < this->neurons.size(); i++) {
        m->setValue(0, i, this->activatedNeurons.at(i));
    }
    return m;
}

Matrix *Layer::matrixifyDerivedValues() {
    auto *m = new Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < this->neurons.size(); i++) {
        m->setValue(0, i, this->derivedNeurons.at(i));
    }
    return m;
}

double generateRandomNumber() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-.0001, .0001);

    return (double) rand()/(double)RAND_MAX;
}
