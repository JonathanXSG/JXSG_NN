#include "../../headers/utils/Misc.hpp"

std::vector< std::vector<double> > utils::Misc::fetchData(std::string path) {
  std::vector< std::vector<double> > data;

  std::ifstream infile(path);

  std::string line;
  while(getline(infile, line)) {
    std::vector<double>  dRow;
    std::string          tok;
    std::stringstream    ss(line);

    while(getline(ss, tok, ',')) {
      dRow.push_back(stof(tok));
    }

    data.push_back(dRow);
  }

  return data;
}

void utils::Misc::printSyntax() {
  std::cout << "Syntax:" << std::endl;
  std::cout << "train [configFile]" << std::endl;
}

ANNConfig utils::Misc::buildConfig(json configObject) {
  ANNConfig config;

  std::vector<int> topology   = configObject["topology"];
  double bias                 = configObject["bias"];
  double learningRate         = configObject["learningRate"];
  double momentum             = configObject["momentum"];
  int epoch                   = configObject["epoch"];
  NN_ACTIVATION hActivation  = configObject["hActivation"];
  NN_ACTIVATION oActivation  = configObject["oActivation"];
  GRADIENT_DESCENT gradDesc   = configObject["gradientDescent"];
  std::cout << "here" << std::endl;
  int batch                   = configObject["batch"];
  std::string trainingFile    = configObject["trainingFile"];
  std::string labelsFile      = configObject["labelsFile"];
  std::string weightsFile     = configObject["weightsFile"];
  
  config.topology         = topology;
  config.bias             = bias;
  config.learningRate     = learningRate;
  config.momentum         = momentum;
  config.epoch            = epoch;
  config.hActivation      = hActivation;
  config.oActivation      = oActivation;
  config.gradientDescent  = gradDesc;
  config.batch            = batch;
  config.trainingFile     = trainingFile;
  config.labelsFile       = labelsFile;
  config.weightsFile      = weightsFile;

  return config;
}
