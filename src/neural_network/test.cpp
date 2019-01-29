#include "../../headers/NeuralNetwork.hpp"
#include <chrono>

void NeuralNetwork::test(
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
    report << "Sample," << " Correct," << " Prediction," << " Target," << " Loss/Cost, " << " Accuracy" << "\n";
    report << std::boolalpha;
    std::cout << std::boolalpha;

    double correctPredictions = 0;

    for (unsigned tIndex = 0; tIndex < labelData.size(); tIndex++) {

        this->setCurrentInput(trainingData.at(tIndex));
        this->setCurrentTarget(labelData.at(tIndex));
        this->feedForward();
        this->setErrors();

        int target = std::distance(labelData.at(tIndex).begin(),
                std::max_element(labelData.at(tIndex).begin(), labelData.at(tIndex).end()));
        int prediction = std::distance(getActivatedNeurons(topologySize - 1)->begin(),
                                   std::max_element(getActivatedNeurons(topologySize - 1)->begin(),
                                                    getActivatedNeurons(topologySize - 1)->end()));
        correctPredictions += (prediction == target)? 1:0;

        report << tIndex << ", " << (prediction == target) << "," << prediction<< ","
                << target<< "," << error << "," << (correctPredictions/(tIndex+1)) << std::endl;
        std::cout << tIndex << ", " << (prediction == target) << "," << prediction<< ","
               << target<< "," << error << "," << (correctPredictions/(tIndex+1)) << std::endl;

//        if (tIndex % (labelData.size() / 10) == 0)
//            std::cout << "*";

    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::endl << "Done!" << std::endl;
    report.close();

}

