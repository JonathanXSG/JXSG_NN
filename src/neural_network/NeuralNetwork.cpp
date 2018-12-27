#include "../../headers/NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(ANNConfig config) {
  this->config        = config;
  this->topology      = config.topology;
  this->topologySize  = config.topology.size();
  this->learningRate  = config.learningRate;
  this->momentum      = config.momentum;
  this->bias          = config.bias;

  this->hiddenActivationType  = config.hActivation;
  this->outputActivationType  = config.oActivation;
  this->costFunctionType      = config.cost;

// Initlializing all layers
  for(int i = 0; i < topologySize; i++) {
    // Initlializing Hidden layers
    if(i > 0 && i < (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i),HIDDEN_FULLYCONNECTED, this->hiddenActivationType);
      this->layers.push_back(l);
    } 
    // Initializing Output layer
    else if(i == (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), OUTPUT, this->outputActivationType);
      this->layers.push_back(l);
    } 
    // Initializing Input Layer
    else {
      Layer *l  = new Layer(topology.at(i), INPUT);
      this->layers.push_back(l);
    }
  }

  // Initializing weights 
  for(int i = 0; i < (topologySize - 1); i++) {
    // Weight matrix with rows = number of Neurons in current layer
    // and columns = number of Neurons in next layer 
    Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);

    this->weightMatrices.push_back(m);
  }

  // Initialize empty errors
  for(int i = 0; i < topology.at(topology.size() - 1); i++) {
    errors.push_back(0.00);
    derivedErrors.push_back(0.00);
  }

  this->error = 0.00;
}

void NeuralNetwork::saveWeights(std::string filename) {
  json j  = {};

  std::vector< std::vector< std::vector<double> > > weightSet;

  for(int i = 0; i < this->weightMatrices.size(); i++) {
    weightSet.push_back(this->weightMatrices.at(i)->getValues());
  }

  j["weights"]      = weightSet;
  j["topology"]     = this->topology;
  j["learningRate"] = this->learningRate;
  j["momentum"]     = this->momentum;
  j["bias"]         = this->bias;

  std::ofstream o(filename);
  o << std::setw(4) << j << std::endl;
}

void NeuralNetwork::setCurrentInput(std::vector<double> input) {
  this->input = input;

  for(int i = 0; i < input.size(); i++) {
    this->layers.at(0)->setVal(i, input.at(i));
  }
}
