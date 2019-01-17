#include "../../headers/NeuralNetwork.hpp"
#include "../../headers/utils/Math.hpp"

void NeuralNetwork::feedForward() {
  Matrix *leftNeurons;  // Matrix of neurons to the left
  Matrix *leftWeights;  // Matrix of weights to the right of layer
  Matrix *rightNeurons;  // Matrix of neurons to the next layer

  for(int i = 0; i < (this->topologySize - 1); i++) {
    // If it's not the input layer, get the activated values
    if(i != 0) {
      leftNeurons = this->getActivatedNeuronMatrix(i);
    }
    // If it's the input layer, get the non-activated values
    else{
      leftNeurons = this->getNeuronMatrix(i);
    }
    leftWeights = this->getWeightMatrix(i);
    rightNeurons = new Matrix(
          leftNeurons->getRows(),
          leftWeights->getColumns(),
          false
        );

    // Here we are basically multiplying the Neurons from the previous layer
    // to the weights that go to each right neuron, them sum them
    // rightNeuron_x =  Sum ( leftNeuron_y * weight_x-y )
    utils::Math::multiplyMatrix(leftNeurons, leftWeights, rightNeurons);

    // Saving the results (adding the bias of the layer) to the next Layer's neuron values
    // The activated values will also be calculated and saved
    for(int rightIndex = 0; rightIndex < rightNeurons->getColumns(); rightIndex++) {
      this->setNeuronValue(i + 1, rightIndex, rightNeurons->getValue(0, rightIndex) + this->bias);
    }
    this->layers.at(i+1)->activate();
    this->layers.at(i+1)->derive();

    delete leftNeurons;
    delete leftWeights;
    delete rightNeurons;
  }
}
