#include "../../headers/utils/Math.hpp"

void utils::Math::multiplyMatrix(Matrix *a, Matrix *b, Matrix *c) {
  for(int i = 0; i < a->getRows(); i++) {
    for(int j = 0; j < b->getColumns(); j++) {
      for(int k = 0; k < b->getRows(); k++) {
        double p      = a->getValue(i, k) * b->getValue(k, j);
        double newVal = c->getValue(i, j) + p;
        c->setValue(i, j, newVal);
      }

      c->setValue(i, j, c->getValue(i, j));
    } 
  }
}
