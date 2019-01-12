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
    static std::vector< std::vector<double> > fetchData(std::string path) {
      std::vector< std::vector<double> > data;

      std::ifstream infile(path);

      std::string line;
      while(getline(infile, line)) {
        std::vector<double>  dRow;
        std::string          tok;
        std::stringstream    ss(line);

        while(getline(ss, tok, ',')) {
          dRow.push_back(stof(tok));
        }

        data.push_back(dRow);
      }

      return data;
    }

    static void printSyntax() {
      std::cout << "Syntax:" << std::endl;
      std::cout << "train [configFile]" << std::endl;
    }

    static ANNConfig buildConfig(json configObject) {
      ANNConfig config;

      std::vector<int> topology   = configObject["topology"];
      double bias                 = configObject["bias"];
      double learningRate         = configObject["learningRate"];
      double momentum             = configObject["momentum"];
      int epoch                   = configObject["epoch"];
      NN_ACTIVATION hActivation  = configObject["hActivation"];
      NN_ACTIVATION oActivation  = configObject["oActivation"];
      GRADIENT_DESCENT gradDesc   = configObject["gradientDescent"];
      std::cout << "here" << std::endl;
      int batch                   = configObject["batch"];
      std::string trainingFile    = configObject["trainingFile"];
      std::string labelsFile      = configObject["labelsFile"];
      std::string weightsFile     = configObject["weightsFile"];
      
      config.topology         = topology;
      config.bias             = bias;
      config.learningRate     = learningRate;
      config.momentum         = momentum;
      config.epoch            = epoch;
      config.hActivation      = hActivation;
      config.oActivation      = oActivation;
      config.gradientDescent  = gradDesc;
      config.batch            = batch;
      config.trainingFile     = trainingFile;
      config.labelsFile       = labelsFile;
      config.weightsFile      = weightsFile;

      return config;
    }

    static void readMNIST(std::vector<std::vector<double>>& dataset,
                          std::vector<std::vector<double>> &labels,
                          std::vector<std::vector<double>>& testDataset,
                          std::vector<std::vector<double>> &testLabels){
      std::string base_dir = "../data/";
      std::string img_path = base_dir + "train-images-idx3-ubyte";
      std::string label_path = base_dir + "train-labels-idx1-ubyte";
      std::string test_img_path = base_dir + "t10k-images.idx3-ubyte";
      std::string test_label_path = base_dir + "t10k-labels-idx1-ubyte";

      read_Mnist(img_path, dataset);
      read_Mnist_Label(label_path, labels);
      read_Mnist(test_img_path, testDataset);
      read_Mnist_Label(test_label_path, testLabels);
    }

    static int ReverseInt (int i){
      unsigned char ch1, ch2, ch3, ch4;
      ch1 = i & 255;
      ch2 = (i >> 8) & 255;
      ch3 = (i >> 16) & 255;
      ch4 = (i >> 24) & 255;
      return ((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }
    
    static void read_Mnist(std::string filename, std::vector<std::vector<double> > &vec){
       std:: ifstream file (filename, std::ios::binary);
        if (file.is_open()){
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;

            file.read((char*) &magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            file.read((char*) &number_of_images,sizeof(number_of_images));
            number_of_images = ReverseInt(number_of_images);
            file.read((char*) &n_rows, sizeof(n_rows));
            n_rows = ReverseInt(n_rows);
            file.read((char*) &n_cols, sizeof(n_cols));
            n_cols = ReverseInt(n_cols);
            for(int i = 0; i < number_of_images; ++i){
                std::vector<double> tp;
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.push_back((double)temp);
                    }
                }
                vec.push_back(tp);
            }
        }
    }
    static void read_Mnist_Label(std::string filename, std::vector<std::vector<double>> &vec){
        std::ifstream file (filename, std::ios::binary);
        if (file.is_open()){
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read((char*) &magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            file.read((char*) &number_of_images,sizeof(number_of_images));
            number_of_images = ReverseInt(number_of_images);
            for(int i = 0; i < number_of_images; ++i){
                unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
                vec.at(i).at(((int)temp)-1) = 1.0;
            }
        }
    }
  };
}

#endif
