#ifndef _MISC_HPP_
#define _MISC_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>
#include "../../headers/NeuralNetwork.hpp"
#include "../json.hpp"

namespace utils {
    class Misc {
    public:
        static void fetchData(const std::string &path, std::vector<std::vector<double>> &data) {
            std::string MNIST_Train = "../data/train-images.idx3-ubyte";
            std::string MNIST_TrainLabels = "../data/train-labels.idx1-ubyte";
            std::string MNIST_Test = "../data/t10k-images.idx3-ubyte";
            std::string MNIST_TestLabels = "../data/t10k-labels.idx1-ubyte";
            if((path == MNIST_Train) || (path == MNIST_Test)){
                read_Mnist(path, data);
            }
            else if((path == MNIST_TrainLabels) || (path == MNIST_TestLabels)){
                read_Mnist_Label(path, data);
            }
            else{
                std::ifstream infile(path);

                std::string line;
                while (getline(infile, line)) {
                    std::vector<double> dRow;
                    std::string tok;
                    std::stringstream ss(line);

                    while (getline(ss, tok, ',')) {
                        dRow.push_back(stof(tok));
                    }

                    data.push_back(dRow);
                }
            }
        }

        static void printSyntax() {
            std::cout << "Syntax:" << std::endl;
            std::cout << "JXSG_NN [configFile] [--test , --train]" << std::endl;
        }

        static void printMatrix(std::vector<std::vector<double>> matrix) {
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[0].size(); j++) {
                    std::cout << std::setprecision(3) << matrix[i][j] << "\t";
                }
                std::cout << std::endl;
            }
        }

        static ANNConfig buildConfig(json configObject) {
            ANNConfig config;

            std::vector<unsigned> topology  = configObject["topology"];
            double bias                     = configObject["bias"];
            double learningRate             = configObject["learningRate"];
            double momentum                 = configObject["momentum"];
            int epochs                      = configObject["epochs"];
            int batch                       = configObject["batch"];
            int iterations                  = configObject["iterations"];
            NN_ACTIVATION hActivation       = configObject["hActivation"];
            NN_ACTIVATION oActivation       = configObject["oActivation"];
            GRADIENT_DESCENT gradDesc       = configObject["gradientDescent"];
            NN_COST costFunction            = configObject["costFunction"];
            std::string dataFile            = configObject["dataFile"];
            std::string labelsFile          = configObject["labelsFile"];
            std::string weightsFile         = configObject["weightsFile"];
            std::string reportFile          = configObject["reportFile"];

            config.topology = topology;
            config.bias = bias;
            config.learningRate = learningRate;
            config.momentum = momentum;
            config.batch = batch;
            config.epochs = epochs;
            config.iterations = iterations;
            config.hActivation = hActivation;
            config.oActivation = oActivation;
            config.gradientDescent = gradDesc;
            config.costFunction = costFunction;
            config.dataFile = dataFile;
            config.labelsFile = labelsFile;
            config.weightsFile = weightsFile;
            config.reportFile = reportFile;

            return config;
        }

        static void printHeader(ANNConfig config){
            std::cout << "|*************************************************|" << std::endl
                      << "|*****       STARTING JXSG_NN TRAINING       *****|" << std::endl
                      << "|*************************************************|" << std::endl
                      << "Topology: \t\t[ ";
            for (auto t : config.topology)
                std::cout << t << " ";
            std::cout << "]" << std::endl
                      << "Learning rate:\t\t" << config.learningRate << std::endl
                      << "Momentum:\t\t" << config.momentum << std::endl
                      << "Iterations:\t\t" << config.iterations << std::endl
                      << "Epochs:\t\t\t" << config.epochs << std::endl
                      << "Batch:\t\t\t" << config.batch << std::endl
                      << "Hidden Activation:\t" << config.hActivation << std::endl
                      << "Output Activation:\t" << config.oActivation << std::endl
                      << "Gradient Descent:\t" << config.gradientDescent << std::endl
                      << std::endl
                      << "Training data file:\t" << config.dataFile << std::endl
                      << "Training label file:\t" << config.labelsFile << std::endl;
        }

        static int ReverseInt(int i) {
            unsigned char ch1, ch2, ch3, ch4;
            ch1 = i & 255;
            ch2 = (i >> 8) & 255;
            ch3 = (i >> 16) & 255;
            ch4 = (i >> 24) & 255;
            return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
        }

        static void read_Mnist(const std::string &filename, std::vector<std::vector<double>> &vec) {
            std::ifstream file(filename, std::ios::binary);
            if (file.is_open()) {
                int magic_number = 0;
                int number_of_images = 0;
                int n_rows = 0;
                int n_cols = 0;

                file.read((char *) &magic_number, sizeof(magic_number));
                magic_number = ReverseInt(magic_number);
                file.read((char *) &number_of_images, sizeof(number_of_images));
                number_of_images = ReverseInt(number_of_images);
                file.read((char *) &n_rows, sizeof(n_rows));
                n_rows = ReverseInt(n_rows);
                file.read((char *) &n_cols, sizeof(n_cols));
                n_cols = ReverseInt(n_cols);
                vec.reserve(number_of_images);

                for (unsigned i = 0; i < number_of_images; ++i) {
                    std::vector<double> tp;
                    tp.reserve(static_cast<unsigned int>(n_rows * n_cols));
                    for (int r = 0; r < n_rows; ++r) {
                        for (int c = 0; c < n_cols; ++c) {
                            unsigned char temp = 0;
                            file.read((char *) &temp, sizeof(temp));
                            if (temp > 0) {
                                tp.emplace_back(1.0);
                            } else {
                                tp.emplace_back(0);
                            }
//                            tp.emplace_back((double) temp/255);
                        }
                    }
                    vec.emplace_back(tp);
                }
            }
        }

        static void read_Mnist_Label(const std::string &filename, std::vector<std::vector<double>> &vec) {
            std::ifstream file(filename, std::ios::binary);
            if (file.is_open()) {
                int magic_number = 0;
                int number_of_labels = 0;
                file.read((char *) &magic_number, sizeof(magic_number));
                magic_number = ReverseInt(magic_number);
                file.read((char *) &number_of_labels, sizeof(number_of_labels));
                number_of_labels = ReverseInt(number_of_labels);
                vec.reserve(number_of_labels);

                for (unsigned i = 0; i < number_of_labels; ++i) {
                    unsigned char temp = 0;
                    file.read((char *) &temp, sizeof(temp));
                    std::vector<double> vect(10, 0.0);
                    vect.at((unsigned  int)temp) = 1.0;
                    vec.emplace_back(vect);
                }
            }
        }
    };
}

#endif
