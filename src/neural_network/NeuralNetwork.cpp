#include "../../headers/NeuralNetwork.hpp"
#include <iomanip>

NeuralNetwork::NeuralNetwork(ANNConfig config) {
    this->config = config;
    this->topology = config.topology;
    this->topologySize = config.topology.size();
    this->learningRate = config.learningRate;
    this->momentum = config.momentum;
    this->bias = config.bias;

    this->hiddenActivationType = config.hActivation;
    this->outputActivationType = config.oActivation;
    this->costFunctionType = config.costFunction;
    this->gradientDescent = config.gradientDescent;

    // Initializing all layers
    this->layers.reserve(topologySize);
    for (unsigned i = 0; i < topologySize; i++) {
        // Initializing Hidden layers
        if (i > 0 && i < (topologySize - 1)) {
            this->layers.emplace_back(new Layer(topology.at(i), HIDDEN_FULLYCONNECTED, this->hiddenActivationType));
            this->layers.at(i)->activate();
            this->layers.at(i)->derive();
        }
        // Initializing Output layer
        else if (i == (topologySize - 1)) {
            this->layers.emplace_back(new Layer(topology.at(i), OUTPUT, this->outputActivationType));
            this->layers.at(i)->activate();
            this->layers.at(i)->derive();
        }
        // Initializing Input Layer
        else {
            this->layers.emplace_back(new Layer(topology.at(i), INPUT));
        }

    }

    // Initializing weights
    this->deltaMatrices.reserve(topologySize-1);
    this->weightMatrices.reserve(topologySize-1);
    for (unsigned i = 0; i < (topologySize - 1); i++) {
        // Weight matrix with rows = number of Neurons in current layer
        // and columns = number of Neurons in next layer
        this->deltaMatrices.emplace_back( new Matrix(topology.at(i), topology.at(i + 1), false));
        this->weightMatrices.emplace_back(new Matrix(topology.at(i), topology.at(i + 1), true));
    }

    // Initialize empty errors
    errors.assign(this->topology.at(topology.size() - 1), 0.00);
    derivedErrors.assign(this->topology.at(topology.size() - 1), 0.00);
    this->error = 0.00;
    this->target = new std::vector<double>(0);
}

void NeuralNetwork::saveWeights(std::string filename) {
    json j = {};

    std::vector<std::vector<std::vector<double> > > weightSet;
    weightSet.reserve(this->weightMatrices.size());

    for (auto &weightMatrix : this->weightMatrices) {
        weightSet.emplace_back(weightMatrix->getValues());
    }

    j["weights"]        = weightSet;
    j["topology"]       = this->topology;
    j["learningRate"]   = this->learningRate;
    j["momentum"]       = this->momentum;
    j["bias"]           = this->bias;
    j["hActivation"]    = this->hiddenActivationType;
    j["oActivation"]    = this->outputActivationType;
    j["reportFile"]     = this->config.reportFile;

    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;
}

void NeuralNetwork::loadWeights(std::string filename) {
    std::ifstream stream(filename);
    json wightFile;
    stream >> wightFile;

    std::vector<std::vector<std::vector<double> > > temp = wightFile["weights"];

    for (unsigned i = 0; i < this->weightMatrices.size(); i++) {
        for (unsigned r = 0; r < this->weightMatrices.at(i)->getRows(); r++) {
            for (unsigned c = 0; c < this->weightMatrices.at(i)->getColumns(); c++) {
                this->weightMatrices.at(i)->at(r, c) = temp.at(i).at(r).at(c);
            }
        }
    }
}

void NeuralNetwork::setCurrentInput(std::vector<double>& input) {
    this->layers.at(0)->setNeurons(&input);
}

void NeuralNetwork::setCurrentTarget(std::vector<double>& target) {
    this->target = &target;
};

