#include "../headers/Matrix.hpp"

Matrix *Matrix::transpose() {
  Matrix *m = new Matrix(this->columns, this->rows, false);

  for(int i = 0; i < this->rows; i++) {
    for(int j = 0; j < this->columns; j++) {
      m->setValue(j, i, this->getValue(i, j));
    }
  }

  return m;
}

Matrix *Matrix::copy() {
  Matrix *m = new Matrix(this->rows, this->columns, false);

  for(int i = 0; i < this->rows; i++) {
    for(int j = 0; j < this->columns; j++) {
      m->setValue(i, j, this->getValue(i, j));
    }
  }

  return m;
}

Matrix::Matrix(int rows, int columns, bool isRandom) {
  this->rows = rows;
  this->columns = columns;

  for(int i = 0; i < rows; i++) {
    std::vector<double> colValues;

    for(int j = 0; j < columns; j++) {
      double r = isRandom == true ? this->generateRandomNumber() : 0.00;
      colValues.push_back(r);
    }

    this->values.push_back(colValues);
  }
}

double Matrix::generateRandomNumber() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-.0001, .0001);

  return dis(gen);
}

void Matrix::printToConsole() {
  for(int i = 0; i < this->rows; i++) {
    for(int j = 0; j < this->columns; j++) {
      std::cout << this->values.at(i).at(j) << "\t";
    }
    std::cout << std::endl;
  }
}

