#ifndef NN_CONVLAYER_HPP
#define NN_CONVLAYER_HPP


#include "Layer.hpp"

class ConvLayer : public Layer{

    ConvLayer(unsigned index, unsigned size, unsigned kernel, unsigned field, unsigned stride, unsigned padding,
            LayerType layerType, ActivationFunc activationType = RELU, double bias = 0);

    void feedForward(const Layer *prevLayer, const Matrix *weights);

    void backPropagation(const Layer *prevLayer, const Matrix *weights);

private :
    unsigned kernel;
    unsigned field;
    unsigned stride;
    unsigned padding;

};


#endif //NN_CONVLAYER_HPP
