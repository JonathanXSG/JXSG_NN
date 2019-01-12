#include "../../headers/NeuralNetwork.hpp"

void NeuralNetwork::train(
        std::vector<std::vector<double>> trainingData,
        std::vector<std::vector<double>> labelData
) {

    for (int i = 0; i < this->config.epoch; i++) {
        for (int tIndex = 0; tIndex < trainingData.size(); tIndex++) {
            std::vector<double> input = trainingData.at(tIndex);
            std::vector<double> target = labelData.at(tIndex);

            this->setCurrentInput(input);
            this->setCurrentTarget(target);
            // this->printToConsole();
//            std::cout << "feed forward" << std::endl;
            this->feedForward();
//       this->printToConsole();
//            std::cout << "errors" << std::endl;
            this->setErrors();


//            std::cout << "back prop" << std::endl;
            if (gradientDescent == STOCHASTIC)
                this->backPropagation();
            else if (gradientDescent == MINI_BATCH && i % this->config.batch == 0)
                this->backPropagation();
            else if (gradientDescent == BATCH && (this->config.epoch - 1) == i)
                this->backPropagation();
        }
        std::cout << "Epoch: " << i + 1 << "Loss/Cost : " << this->error << ""
                  << std::endl;
    }

}
