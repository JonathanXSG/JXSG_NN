#include "../../headers/NeuralNetwork.hpp"

void NeuralNetwork::train(
  std::vector< std::vector<double>> trainingData, 
  std::vector< std::vector<double>> labelData
) {

  for(int i = 0; i < this->config.epoch; i++) {
    for(int tIndex = 0; tIndex < trainingData.size(); tIndex++) {
      std::vector<double> input    = trainingData.at(tIndex);
      std::vector<double> target   = labelData.at(tIndex);

      this ->setCurrentInput(input);
      this->setCurrentTarget(target);
      // std::cout << "initial" << std::endl;
      // this->printToConsole();
      this->feedForward();
      //       std::cout << "feed forward" << std::endl;
      // this->printToConsole();
      this->setErrors();

      if(gradientDescent == STOCHASTIC)
        this->backPropagation();
       else if(gradientDescent == MINI_BATCH && i % this->config.batch == 0)
        this->backPropagation();
      else if(gradientDescent == BATCH && (this->config.epoch-1) == i)
        this->backPropagation();
    }
    std::cout << "Epoch: " << i+1 << "Loss/Cost : " << this->error << "" 
      << std::endl;
  }
  
}
