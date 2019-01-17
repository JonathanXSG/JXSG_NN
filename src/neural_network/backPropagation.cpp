#include "../../headers/NeuralNetwork.hpp"
#include "../../headers/utils/Math.hpp"

void NeuralNetwork::backPropagation() {
  std::vector<Matrix *> newWeights;
  Matrix *deltaWeights;
  Matrix *gradients;
  Matrix *derivedValues;
  Matrix *gradientsTransposed;
  Matrix *zActivatedValues;
  Matrix *tempNewWeights;
  Matrix *prevGradients;
  Matrix *transposedPrevWeights;
  Matrix *hiddenDerivedValues;
  Matrix *transposedHidden;

  unsigned indexOutputLayer  = this->topology.size() - 1;

  // From output to last hidden layer
  gradients = new Matrix(
                1,
                this->topology.at(indexOutputLayer),
                false
              );

  derivedValues = this->layers.at(indexOutputLayer)->matrixifyDerivedValues();

  // Calculating gradients 
  // gradient = dError * dNeuronValue
  for(int i = 0; i < this->topology.at(indexOutputLayer); i++) {
    double derivedError  = this->derivedErrors.at(i);
    double derivedValue  = derivedValues->getValue(0, i);
    gradients->setValue(0, i, derivedError * derivedValue);
  }


  // DeltaWeights = Gt * Z
  // Where:  Gt is the transposed Gradients matrix
  //         Z is the matrix of activated values in the last hidden layer
  gradientsTransposed = gradients->transpose();
  zActivatedValues      = this->layers.at(indexOutputLayer - 1)->matrixifyActivatedValues();

  deltaWeights  = new Matrix(
                    gradientsTransposed->getRows(),
                    zActivatedValues->getColumns(),
                    false
                  );    

  ::utils::Math::multiplyMatrix(gradientsTransposed, zActivatedValues, deltaWeights);

  // Calculating the new weights
  tempNewWeights  = new Matrix(
                      this->topology.at(indexOutputLayer - 1),
                      this->topology.at(indexOutputLayer),
                      false
                    );

  for(int row = 0; row < this->topology.at(indexOutputLayer - 1); row++) {
    for(int col = 0; col < this->topology.at(indexOutputLayer); col++) {
      double originalWeight  = this->weightMatrices.at(indexOutputLayer - 1)->getValue(row, col);
      double deltaWeight     = deltaWeights->getValue(col, row);

      originalWeight = this->momentum * originalWeight;
      deltaWeight    = this->learningRate * deltaWeight;
      
      tempNewWeights->setValue(row, col, (originalWeight - deltaWeight));
    }
  }

  newWeights.push_back(new Matrix(*tempNewWeights));

  delete gradientsTransposed;
  delete zActivatedValues;
  delete tempNewWeights;
  delete deltaWeights;
  delete derivedValues;

  ///////////////////////////
  // From the last hidden layer to input layer
  for(int i = (indexOutputLayer - 1); i > 0; i--) {
    prevGradients  = new Matrix(*gradients);
    delete gradients;

    transposedPrevWeights  = this->weightMatrices.at(i)->transpose();

    gradients              = new Matrix(
                            prevGradients->getRows(),
                            transposedPrevWeights->getColumns(),
                            false
                          );

    ::utils::Math::multiplyMatrix(prevGradients, transposedPrevWeights, gradients);

    hiddenDerivedValues = this->layers.at(i)->matrixifyDerivedValues();

    for(int colCounter = 0; colCounter < hiddenDerivedValues->getColumns(); colCounter++) {
      double  g = gradients->getValue(0, colCounter) * hiddenDerivedValues->getValue(0, colCounter);
      gradients->setValue(0, colCounter, g);
    }

    if(i == 1) {
      zActivatedValues  = this->layers.at(0)->matrixifyValues();
    } else {
      zActivatedValues  = this->layers.at(i-1)->matrixifyActivatedValues();
    }

    transposedHidden  = zActivatedValues->transpose();

    deltaWeights      = new Matrix(
                          transposedHidden->getRows(),
                          gradients->getColumns(),
                          false
                        );

    ::utils::Math::multiplyMatrix(transposedHidden, gradients, deltaWeights);

    // update weights
    tempNewWeights  = new Matrix(
                        this->weightMatrices.at(i - 1)->getRows(),
                        this->weightMatrices.at(i - 1)->getColumns(),
                        false
                      );

    for(int row = 0; row < tempNewWeights->getRows(); row++) {
      for(int col = 0; col < tempNewWeights->getColumns(); col++) {
        double originalWeight  = this->weightMatrices.at(i - 1)->getValue(row, col);
        double deltaWeight     = deltaWeights->getValue(row, col);

        originalWeight = this->momentum * originalWeight;
        deltaWeight    = this->learningRate * deltaWeight;
        
        tempNewWeights->setValue(row, col, (originalWeight - deltaWeight));
      }
    }

    newWeights.push_back(new Matrix(*tempNewWeights));

    delete prevGradients;
    delete transposedPrevWeights;
    delete hiddenDerivedValues;
    delete zActivatedValues;
    delete transposedHidden;
    delete tempNewWeights;
    delete deltaWeights;
  }

  for (auto &weightMatrix : this->weightMatrices) {
    delete weightMatrix;
  }

  this->weightMatrices.clear();

  reverse(newWeights.begin(), newWeights.end());

  for (auto &newWeight : newWeights) {
    this->weightMatrices.push_back(new Matrix(*newWeight));
    delete newWeight;
  }
}
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
