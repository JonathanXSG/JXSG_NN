#include <iostream>
#include <vector>
#include <cstdio>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <ctime>
#include "../headers/json.hpp"
#include "../headers/NeuralNetwork.hpp"
#include "../headers/utils/Misc.hpp"
#include <chrono>

using json = nlohmann::json;

int main(int argc, char **argv) {
    auto start = std::chrono::high_resolution_clock::now();

    if (argc != 2) {
        utils::Misc::printSyntax();
        exit(-1);
    }

    std::ifstream configFile(argv[1]);
    std::string str((std::istreambuf_iterator<char>(configFile)),
                    std::istreambuf_iterator<char>());

    NeuralNetwork *n = new NeuralNetwork(utils::Misc::buildConfig(json::parse(str)));

    utils::Misc::printHeader(n->config);

    std::vector<std::vector<double>> trainingData;
    std::vector<std::vector<double>> labelData;

    utils::Misc::fetchData(n->config.trainingFile, trainingData);
    utils::Misc::fetchData(n->config.labelsFile, labelData);

    std::cout << "Training Data Size:\t" << trainingData.size() << std::endl;
    std::cout << "Label Data Size:\t" << labelData.size() << std::endl << std::endl;

    n->train(trainingData, labelData);

    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);
    std::stringstream ss;
    ss << "../weights/" << n->config.weightsFile << " "
       << (now->tm_year + 1900) << '-'
       << (now->tm_mon + 1) << '-'
       << now->tm_mday << '-'
       << now->tm_hour << '-'
       << now->tm_min << '-'
       << now->tm_sec << ".json";
    std::string filename = ss.str();

    std::cout << "Done! Writing to " << filename << "..." << std::endl;
    n->saveWeights(filename);
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << "ns\n";
    return 0;
}


