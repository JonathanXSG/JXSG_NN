#include "../../headers/NeuralNetwork.hpp"

void NeuralNetwork::loadWeights(std::string filename) {
  std::ifstream i(filename);
  json jWeights;
  i >> jWeights;

  std::vector< std::vector< std::vector<double> > > temp = jWeights["weights"];

  for(int i = 0; i < this->weightMatrices.size(); i++) {
    for(int r = 0; r < this->weightMatrices.at(i)->getRows(); r++) {
      for(int c = 0; c < this->weightMatrices.at(i)->getColumns(); c++) {
        this->weightMatrices.at(i)->setValue(r, c, temp.at(i).at(r).at(c));
      }
    }
  }
}
