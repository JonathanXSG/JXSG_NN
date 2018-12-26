#ifndef _MISC_HPP_
#define _MISC_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <assert.h>
#include "../../headers/NeuralNetwork.hpp"
#include "../json.hpp"

namespace utils
{
  class Misc
  {
  public:
    static std::vector< std::vector<double> > fetchData(std::string path);
    static void printSyntax();
    static ANNConfig buildConfig(json configObject);
  };
}

#endif
