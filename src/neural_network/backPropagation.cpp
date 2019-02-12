#include "../../headers/NeuralNetwork.hpp"
#include "../../headers/utils/Math.hpp"

void NeuralNetwork::backPropagation() {
    std::vector<std::vector<double>> *gradients;
    std::vector<double> *zActivatedValues;

    unsigned indexOutputLayer = this->topology.size() - 1;

    gradients = new std::vector<std::vector<double>>(
            this->topologySize,
            std::vector<double>());

    for (unsigned i = indexOutputLayer; i > 0; i--) {
        gradients->at(i) = std::vector<double>(
                this->topology.at(i),
                0);

        // *********************************
        // Calculating Gradients from the last hidden layer to input layer
        // *********************************
        if (i != indexOutputLayer){
            // gradient = dError * dNeuronValue
//            #pragma omp parallel for schedule(static, 1) collapse(2)
            for (unsigned r = 0; r < this->topology.at(i); r++) {
                for (unsigned c = 0; c < this->topology.at(i+1); c++) {
                    gradients->at(i).at(r) += (gradients->at(i+1).at(c) * this->weightMatrices.at(i)->at(r, c))
                            * this->getDerivedNeurons(i)->at(r);
                }
            }
        }
        // *********************************
        // Calculating Gradients of output layer
        // *********************************
        else{
            // gradient = dError * dNeuronValue
//            #pragma omp parallel for schedule(static, 1)
            for (unsigned r = 0; r < this->topology.at(indexOutputLayer); r++) {
                gradients->at(indexOutputLayer).at(r) = this->derivedErrors.at(r) *
                                                        this->getDerivedNeurons(indexOutputLayer)->at(r);
            }
        }
    }

    for (unsigned i = indexOutputLayer; i > 0; i--) {
        if (i != 1)
            zActivatedValues = this->getActivatedNeurons(i - 1);
        else
            zActivatedValues = this->getNeurons(i - 1);

        // Calculating the new weights and deltaWeights
//        #pragma omp parallel for schedule(static, 1) collapse(2)
        for (unsigned row = 0; row < this->topology.at(i - 1); row++) {
            for (unsigned col = 0; col < this->topology.at(i); col++) {
                double originalDelta = this->deltaMatrices.at(i - 1)->at(row, col) * this->momentum;
                double newDelta = gradients->at(i).at(col) * zActivatedValues->at(row) * this->learningRate;

                this->deltaMatrices.at(i - 1)->at(row, col) = originalDelta + newDelta;
                this->weightMatrices.at(i - 1)->at(row, col) -= this->deltaMatrices.at(i - 1)->at(row, col);
            }
        }
    }

    for(auto grad : *gradients){
        grad.clear();
    }
    gradients->clear();
    delete gradients;
}

//Old code from the original author
// void NeuralNetwork::backPropagation() {
//   std::vector<Matrix *> newWeights;
//   Matrix *deltaWeights;
//   Matrix *gradients;
//   Matrix *derivedValues;
//   Matrix *zActivatedValues;
//   Matrix *tempNewWeights;
//   Matrix *prevGradients;
//   Matrix *transposedPrevWeights;
//   Matrix *hiddenDerivedValues;
//   Matrix *transposedHidden;

//   int indexOutputLayer  = this->topology.size() - 1;

//   // From output to last hidden layer
//   gradients = new Matrix(
//                 1,
//                 this->topology.at(indexOutputLayer),
//                 false
//               );

//   derivedValues = this->layers.at(indexOutputLayer)->matrixifyDerivedValues;

//   // Calculating gradients 
//   // gradient = dError * dNeuronValue
//   for(int i = 0; i < this->topology.at(indexOutputLayer); i++) {
//     double derivedError  = this->derivedErrors.at(i);
//     double derivedValue  = derivedValues->at(0, i);
//     gradients->setValue(0, i, derivedError * derivedValue);
//   }

//   // DeltaWeights = Gt * Z
//   // Where:  Gt is the transposed Gradients matrix
//   //         Z is the matrix of activated values in the last hidden layer
//   zActivatedValues      = this->layers.at(indexOutputLayer - 1)->matrixifyActivatedValues;

//   deltaWeights  = new Matrix(
//                     zActivatedValues->getRows(),
//                     gradients->getColumns(),
//                     false
//                   );    

//   ::utils::Math::multiplyMatrix(zActivatedValues, gradients, deltaWeights);

//   // Calculating the new weights
//   tempNewWeights  = new Matrix(
//                       this->topology.at(indexOutputLayer - 1),
//                       this->topology.at(indexOutputLayer),
//                       false
//                     );

//   for(int row = 0; row < this->topology.at(indexOutputLayer - 1); row++) {
//     for(int col = 0; col < this->topology.at(indexOutputLayer); col++) {
//       double originalWeight  = this->weightMatrices.at(indexOutputLayer - 1)->at(row, col);
//       // row and col inverted here for the transpose
//       double deltaWeight     = deltaWeights->at(row, col);

//       originalWeight = this->momentum * originalWeight;
//       deltaWeight    = this->learningRate * deltaWeight;

//       tempNewWeights->setValue(row, col, (originalWeight - deltaWeight));
//     }
//   }

//   newWeights.push_back(new Matrix(*tempNewWeights));

//   delete zActivatedValues;
//   delete tempNewWeights;
//   delete deltaWeights;
//   delete derivedValues;

//   ///////////////////////////
//   // From the last hidden layer to input layer
//   for(int i = (indexOutputLayer - 1); i > 0; i--) {
//     prevGradients  = new Matrix(*gradients);
//     delete gradients;

//     transposedPrevWeights  = this->weightMatrices.at(i)->transpose();

//     gradients              = new Matrix(
//                             prevGradients->getRows(),
//                             transposedPrevWeights->getColumns(),
//                             false
//                           );

//     ::utils::Math::multiplyMatrix(prevGradients, transposedPrevWeights, gradients);

//     hiddenDerivedValues = this->layers.at(i)->matrixifyDerivedValues;

//     for(int colCounter = 0; colCounter < hiddenDerivedValues->getColumns(); colCounter++) {
//       double  g = gradients->getValue(0, colCounter) * hiddenDerivedValues->at(0, colCounter);
//       gradients->setValue(0, colCounter, g);
//     }

//     // If we re in the input layer get the un-activated values
//     if(i == 1) {
//       transposedHidden  = this->layers.at(0)->matrixifyValues;
//     } else {
//       transposedHidden  = this->layers.at(i-1)->matrixifyActivatedValues;
//     }

//     deltaWeights      = new Matrix(
//                           transposedHidden->getRows(),
//                           gradients->getColumns(),
//                           false
//                         );

//     ::utils::Math::multiplyMatrix(transposedHidden, gradients, deltaWeights);

//     // update weights
//     tempNewWeights  = new Matrix(
//                         this->weightMatrices.at(i - 1)->getRows(),
//                         this->weightMatrices.at(i - 1)->getColumns(),
//                         false
//                       );

//     for(int row = 0; row < tempNewWeights->getRows(); row++) {
//       for(int col = 0; col < tempNewWeights->getColumns(); col++) {
//         double originalWeight  = this->weightMatrices.at(i - 1)->at(row, col);
//         double deltaWeight     = deltaWeights->at(row, col);

//         originalWeight = this->momentum * originalWeight;
//         deltaWeight    = this->learningRate * deltaWeight;

//         tempNewWeights->setValue(row, col, (originalWeight - deltaWeight));
//       }
//     }

//     newWeights.push_back(new Matrix(*tempNewWeights));

//     delete prevGradients;
//     delete transposedPrevWeights;
//     delete hiddenDerivedValues;
//     delete zActivatedValues;
//     delete transposedHidden;
//     delete tempNewWeights;
//     delete deltaWeights;
//   }

//   for(int i = 0; i < this->weightMatrices.size(); i++) {
//     delete this->weightMatrices[i];
//   }

//   this->weightMatrices.clear();

//   reverse(newWeights.begin(), newWeights.end());

//   for(int i = 0; i < newWeights.size(); i++) {
//     this->weightMatrices.push_back(new Matrix(*newWeights[i]));
//     delete newWeights[i];
//   }

// }
