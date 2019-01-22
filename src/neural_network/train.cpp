#include "../../headers/NeuralNetwork.hpp"
#include <chrono>

void NeuralNetwork::train(
        std::vector<std::vector<double>>& trainingData,
        std::vector<std::vector<double>>& labelData
) {

    for (int i = 0; i < this->config.epoch; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int tIndex = 0; tIndex < 10000; tIndex++) {

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

//            std::cout << tIndex << "   "<< this->error << std::endl;
        }
//        std::cout << target.size() << std::endl;
//        this->layers.at(this->layers.size()-1)->matrixifyActivatedValues()->printToConsole();
//        std::cout << this->layers.at(this->layers.size()-1)->getNeurons().size() << std::endl;

//        std::cout << "Output: ";
//        this->layers.at(this->layers.size()-1)->matrixifyActivatedValues()->printToConsole();
//        std::cout << "\nTarget: ";
//        for (double j : this->target) {
//            std::cout << std::setprecision(5) << j << "\t";
//        }
//        std::cout << std::endl;
//        std::cout << std::endl;
        std::cout << "Epoch: " << i + 1 << "Loss/Cost : " << this->error << ""
                  << std::endl;
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << "ns\n";
    }

}
