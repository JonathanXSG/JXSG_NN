#ifndef NN_FCLAYER_HPP
#define NN_FCLAYER_HPP

#include "Layer.hpp"

class FCLayer : public Layer {

    FCLayer(unsigned index, unsigned size, LayerType layerType, ActivationFunc activationType = RELU,
               double bias = 0);

    void feedForward(const Layer *prevLayer, const Matrix *weights);

    void backPropagation(const Layer *prevLayer, const Matrix *weights);
};

#endif //NN_FCLAYER_HPP
