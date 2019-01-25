#include "../../headers/NeuralNetwork.hpp"
#include "../../headers/utils/Math.hpp"

void NeuralNetwork::backPropagation() {
    std::vector<Matrix *> *newWeights = new std::vector<Matrix *>();
    std::vector<double> *gradients;
    std::vector<double> *zActivatedValues;
    Matrix *tempNewWeights;
    std::vector<double> *prevGradients;

    unsigned indexOutputLayer = this->topology.size() - 1;
    newWeights->reserve(this->weightMatrices.size());

    // *********************************
    // Calculating Gradients of output layer
    // *********************************
    gradients = new std::vector<double>(this->topology.at(indexOutputLayer));

    // Calculating gradients
    // gradient = dError * dNeuronValue
    for (unsigned i = 0; i < this->topology.at(indexOutputLayer); i++) {
        gradients->at(i) = this->derivedErrors.at(i) * this->layers.at(indexOutputLayer)->getDerivedValues()->at(i);
    }

    // Z is the vector of activated values in the last hidden layer
    zActivatedValues = this->layers.at(indexOutputLayer - 1)->getActivatedValues();

    // Calculating the new weights
    tempNewWeights = new Matrix(
            this->topology.at(indexOutputLayer - 1),
            this->topology.at(indexOutputLayer),
            false
    );

    for (unsigned row = 0; row < this->topology.at(indexOutputLayer - 1); row++) {
        for (unsigned col = 0; col < this->topology.at(indexOutputLayer); col++) {
            double originalWeight = this->weightMatrices.at(indexOutputLayer - 1)->getValue(row, col);
            double deltaWeight = gradients->at(col) * zActivatedValues->at(row);

            originalWeight *= this->momentum;
            deltaWeight *= this->learningRate;

            tempNewWeights->setValue(row, col, (originalWeight - deltaWeight));
        }
    }

    newWeights->push_back(tempNewWeights);

    // *********************************
    // Calculating Gradients from the last hidden layer to input layer
    // *********************************
    for (unsigned i = (indexOutputLayer - 1); i > 0; i--) {
        prevGradients = gradients;

        gradients = new std::vector<double>(
                this->weightMatrices.at(i)->getRows()
        );
        double sum;
        for (unsigned r = 0; r < this->weightMatrices.at(i)->getRows(); r++) {
            sum = 0.0;
            for (unsigned c = 0; c < prevGradients->size(); c++) {
                sum += (prevGradients->at(c) * this->weightMatrices.at(i)->getValue(r, c));
            }
            gradients->at(r) = this->layers.at(i)->getDerivedValues()->at(r) * sum;
        }

        if (i == 1) {
            zActivatedValues = this->layers.at(0)->getNeurons();
        } else {
            zActivatedValues = this->layers.at(i - 1)->getActivatedValues();
        }

        // update weights
        tempNewWeights = new Matrix(
                this->weightMatrices.at(i - 1)->getRows(),
                this->weightMatrices.at(i - 1)->getColumns(),
                false
        );

        for (unsigned row = 0; row < tempNewWeights->getRows(); row++) {
            for (unsigned col = 0; col < tempNewWeights->getColumns(); col++) {
                double originalWeight = this->weightMatrices.at(i - 1)->getValue(row, col);
                double deltaWeight = zActivatedValues->at(row) * gradients->at(col);

                originalWeight = this->momentum * originalWeight;
                deltaWeight = this->learningRate * deltaWeight;

                tempNewWeights->setValue(row, col, (originalWeight - deltaWeight));
            }
        }

        newWeights->push_back(tempNewWeights);
        delete prevGradients;
    }

    for (auto &weightMatrix : this->weightMatrices) {
        delete weightMatrix;
    }

    this->weightMatrices.clear();

    reverse(newWeights->begin(), newWeights->end());
    this->weightMatrices = *newWeights;

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
//     double derivedValue  = derivedValues->getValue(0, i);
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
//       double originalWeight  = this->weightMatrices.at(indexOutputLayer - 1)->getValue(row, col);
//       // row and col inverted here for the transpose
//       double deltaWeight     = deltaWeights->getValue(row, col);

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
//       double  g = gradients->getValue(0, colCounter) * hiddenDerivedValues->getValue(0, colCounter);
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
//         double originalWeight  = this->weightMatrices.at(i - 1)->getValue(row, col);
//         double deltaWeight     = deltaWeights->getValue(row, col);

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
