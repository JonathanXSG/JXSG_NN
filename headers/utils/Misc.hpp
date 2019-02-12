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
        static void fetchData(const std::string &path, std::vector<std::vector<double>> &data);

        static void printSyntax();

        static void printMatrix(std::vector<std::vector<double>> matrix);

        static NNConfig buildConfig(json configObject);

        static void printHeader(std::ostream& stream, NNConfig config);

        static int ReverseInt(int i);

        static void read_Mnist(const std::string &filename, std::vector<std::vector<double>> &vec);

        static void read_Mnist_Label(const std::string &filename, std::vector<std::vector<double>> &vec) ;
    };
}

#endif
