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
        static std::vector<std::vector<double> > fetchData(const std::string &path) {
            std::vector<std::vector<double> > data;

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

            return data;
        }

        static void printSyntax() {
            std::cout << "Syntax:" << std::endl;
            std::cout << "JXSG_NN [configFile]" << std::endl;
        }

        static void printMatrix(std::vector<std::vector<double>> matrix) {
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[0].size(); j++) {
                    std::cout << std::setprecision(2) << matrix[i][j] << "\t";
                }
                std::cout << std::endl;
            }
        }

        static ANNConfig buildConfig(json configObject) {
            ANNConfig config;

            std::vector<unsigned> topology = configObject["topology"];
            double bias = configObject["bias"];
            double learningRate = configObject["learningRate"];
            double momentum = configObject["momentum"];
            int epoch = configObject["epoch"];
            NN_ACTIVATION hActivation = configObject["hActivation"];
            NN_ACTIVATION oActivation = configObject["oActivation"];
            GRADIENT_DESCENT gradDesc = configObject["gradientDescent"];
            int batch = configObject["batch"];
            std::string trainingFile = configObject["trainingFile"];
            std::string labelsFile = configObject["labelsFile"];
            std::string weightsFile = configObject["weightsFile"];

            config.topology = topology;
            config.bias = bias;
            config.learningRate = learningRate;
            config.momentum = momentum;
            config.epoch = epoch;
            config.hActivation = hActivation;
            config.oActivation = oActivation;
            config.gradientDescent = gradDesc;
            config.batch = batch;
            config.trainingFile = trainingFile;
            config.labelsFile = labelsFile;
            config.weightsFile = weightsFile;

            return config;
        }

        static void readMNIST(std::vector<std::vector<double>> &dataset,
                              std::vector<std::vector<double>> &labels,
                              std::vector<std::vector<double>> &testDataset,
                              std::vector<std::vector<double>> &testLabels) {
            std::string img_path = "../data/train-images.idx3-ubyte";
            std::string label_path = "../data/train-labels.idx1-ubyte";
            std::string test_img_path = "../data/t10k-images.idx3-ubyte";
            std::string test_label_path = "../data/t10k-labels.idx1-ubyte";

            read_Mnist(img_path, dataset);
            std::cout << dataset.size() << "\t";
            read_Mnist_Label(label_path, labels);
            std::cout << labels.size() << "\t";
            read_Mnist(test_img_path, testDataset);
            std::cout << testDataset.size() << "\t";
            read_Mnist_Label(test_label_path, testLabels);
            std::cout << testLabels.size() << std::endl;
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
                int number_of_images = 0;
                file.read((char *) &magic_number, sizeof(magic_number));
                magic_number = ReverseInt(magic_number);
                file.read((char *) &number_of_images, sizeof(number_of_images));
                number_of_images = ReverseInt(number_of_images);
                for (unsigned i = 0; i < number_of_images; ++i) {
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
