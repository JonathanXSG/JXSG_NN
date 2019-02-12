#include "../../headers/NeuralNetwork.hpp"
#include "../../headers/utils/Misc.hpp"
#include <chrono>

void NeuralNetwork::train(
        std::vector<std::vector<double>> &trainingData,
        std::vector<std::vector<double>> &labelData
) {
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);
    std::stringstream ss;
    ss << "../reports/" << this->config.reportFile << " "
       << (now->tm_year + 1900) << '-'
       << (now->tm_mon + 1) << '-'
       << now->tm_mday << '-'
       << now->tm_hour << '-'
       << now->tm_min << '-'
       << now->tm_sec
       << ".csv";
    std::ofstream report(ss.str());
    utils::Misc::printHeader(report, this->config);

    report << "Epoch," << " Loss/Cost" << "\n";
    std::cout << "\t\t" << "Epoch: " << "\t\t" << "Loss/Cost: " << std::endl;

    for (int i = 0; i < this->config.epochs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned tIndex = 0; tIndex < labelData.size(); tIndex++) {

            this->setCurrentInput(trainingData.at(tIndex));
            this->setCurrentTarget(labelData.at(tIndex));
            for(int iterations = 0; iterations < this->config.iterations; iterations++) {
                this->feedForward();
                this->setErrors();

                if (gradientDescent == Stochastic)
                    this->backPropagation();
                else if (gradientDescent == MiniBatch && i % this->config.batch == 0)
                    this->backPropagation();
                else if (gradientDescent == Batch && (this->config.epochs - 1) == i)
                    this->backPropagation();
            }
            if (tIndex % (labelData.size() / 10) == 0) {
                std::cout << "*";
//                std::cout << "Sample: " << tIndex << " Loss/Cost : " << this->error << ""
//                          << std::endl;
            }
        }
        auto finish = std::chrono::high_resolution_clock::now();

        report << i+1 << ", " << this->error << "\n";
        std::cout << "\t" << i + 1 << " / " << this->config.epochs << "\t\t"
            << this->error << "\t"
            <<std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
            << " us" << std::endl;
    }
    report.close();

}
