cmake_minimum_required(VERSION 3.7)
project(MP_NN)

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

add_executable(
        MP_NN
        src/main.cpp
        src/Layer.cpp
        src/neural_network/NeuralNetwork.cpp
        src/neural_network/setErrors.cpp
        src/neural_network/feedForward.cpp
        src/neural_network/backPropagation.cpp
        src/neural_network/train.cpp
        src/neural_network/test.cpp src/utils/Misc.cpp)
