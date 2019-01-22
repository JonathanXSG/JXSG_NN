#include "../../headers/NeuralNetwork.hpp"
#include "../../headers/utils/Math.hpp"

void NeuralNetwork::feedForward() {
  std::vector<double> *leftNeurons;   // Matrix of neurons to the left
  Matrix *leftWeights;                // Matrix of weights to the right of layer
  std::vector<double> *rightNeurons;  // Matrix of neurons to the next layer

  for(unsigned i = 0; i < (this->topologySize - 1); i++) {
    // If it's not the input layer, get the activated values
    if(i != 0) {
      leftNeurons = this->getActivatedValues(i);
    }
    // If it's the input layer, get the non-activated values
    else{
      leftNeurons = this->getNeurons(i);
    }
    leftWeights = this->getWeightMatrix(i);

    rightNeurons = new std::vector<double>(
          leftWeights->getColumns(),
          this->bias
        );

    // Here we are basically multiplying the Neurons from the previous layer
    // to the weights that go to each right neuron, them sum them
    // rightNeuron_x =  Sum ( leftNeuron_y * weight_x-y )
//    utils::Math::multiplyMatrix(leftNeurons, leftWeights, rightNeurons);

    for(unsigned r = 0; r < leftWeights->getRows(); r++){
        for(unsigned c = 0; c < leftWeights->getColumns(); c++){
            rightNeurons->at(c) += leftNeurons->at(r) * leftWeights->getValue(r,c);
        }
    }

    // Saving the results (adding the bias of the layer) to the next Layer's neuron values
    // The activated values will also be calculated and saved
//    for(int rightIndex = 0; rightIndex < rightNeurons->getColumns(); rightIndex++) {
//      this->setNeuronValue(i + 1, rightIndex, rightNeurons->getValue(0, rightIndex) + this->bias);
//    }
    this->layers.at(i+1)->setNeurons(rightNeurons);
    this->layers.at(i+1)->activate();
    this->layers.at(i+1)->derive();
  }
}
