#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>
#include "../headers/json.hpp"
#include "../headers/NeuralNetwork.hpp"
#include "../headers/utils/Misc.hpp"
#include <chrono>

using json = nlohmann::json;

int main(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();

  if(argc != 2) {
    utils::Misc::printSyntax();
    exit(-1);
  }

  std::ifstream configFile(argv[1]);
  std::string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  NeuralNetwork *n  = new NeuralNetwork(utils::Misc::buildConfig(json::parse(str)));

  std::vector< std::vector<double>> trainingData = utils::Misc::fetchData(n->config.trainingFile);
  std::vector< std::vector<double> > labelData    = utils::Misc::fetchData(n->config.labelsFile);

  std::cout << "Training Data Size: " << trainingData.size() << std::endl;
  std::cout << "Label Data Size: " << labelData.size() << std::endl;

  n->train(trainingData, labelData);

  std::cout << "Done! Writing to " << n->config.weightsFile << "..." << std::endl;
  n->saveWeights(n->config.weightsFile);
    auto finish = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
  return 0;
}


