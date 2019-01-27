#include "../../headers/NeuralNetwork.hpp"
#include <chrono>

void NeuralNetwork::train(
        std::vector<std::vector<double>> &trainingData,
        std::vector<std::vector<double>> &labelData
) {

    for (int i = 0; i < this->config.epoch; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned tIndex = 0; tIndex < labelData.size() / 6; tIndex++) {

            this->setCurrentInput(trainingData.at(tIndex));
            this->setCurrentTarget(labelData.at(tIndex));
//            for(int iterations = 0 ; iterations<20; iterations++){
            this->feedForward();
            this->setErrors();

//                if (gradientDescent == STOCHASTIC)
            this->backPropagation();
//                else if (gradientDescent == MINI_BATCH && i % this->config.batch == 0)
//                    this->backPropagation();
//                else if (gradientDescent == BATCH && (this->config.epoch - 1) == i)
//                    this->backPropagation();
//            }

            if (tIndex % (labelData.size() / 60) == 0) {
                std::cout << "*";
//                std::cout << "Sample: " << tIndex << " Loss/Cost : " << this->error << ""
//                          << std::endl;
            }
        }
        std::cout << std::endl;
        std::cout << "Epoch: " << i + 1 << " Loss/Cost : " << this->error << ""
                  << std::endl;
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << "ns\n";
        std::cout << std::endl;
    }

}
