#include "../../headers/NeuralNetwork.hpp"

void NeuralNetwork::feedForward() {
    std::vector<double> *leftNeurons;   // Matrix of neurons to the left
    Matrix *leftWeights;                // Matrix of weights between leftNeurons and next layer

    for (unsigned i = 0; i < (this->topologySize - 1); i++) {
        // If it's not the input layer, get the activated values
        if (i != 0)
            leftNeurons = this->getActivatedNeurons(i);
        // If it's the input layer, get the non-activated values
        else
            leftNeurons = this->getNeurons(i);

        leftWeights = this->getWeightMatrix(i);

        fill(this->getNeurons(i + 1)->begin(),
                this->getNeurons(i + 1)->end(),
                this->bias);

        // Here we are basically multiplying the Neurons from the previous layer
        // to the weights that go to each right neuron, them sum them
        // rightNeuron_x =  Sum ( leftNeuron_y * weight_x-y )
//        #pragma omp parallel for schedule(static, 1) collapse(2)
        for (unsigned r = 0; r < leftWeights->getHeight(); r++) {
            for (unsigned c = 0; c < leftWeights->getWidth(); c++) {
                this->getNeurons(i+1)->at(c) += (leftNeurons->at(r) * leftWeights->at(r, c));
            }
        }

        this->layers.at(i + 1)->activate();
        this->layers.at(i + 1)->derive();
    }
}
