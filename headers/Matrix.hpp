#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

class Matrix
{
public:
  Matrix(int rows, int columns, bool isRandom); 

  Matrix *transpose();
  Matrix *copy();

  void setValue(int r, int c, double v) { 
    this->values.at(r).at(c) = v; 
  }
  double getValue(int r, int c) { 
    return this->values.at(r).at(c); 
  }

  std::vector< std::vector<double> > getValues() { 
    return this->values; 
  }

  void printToConsole();

  int getRows() { 
    return this->rows; 
  }
  int getColumns() { 
    return this->columns; 
  }

private:
  double generateRandomNumber();

  int rows;
  int columns;

  std::vector< std::vector<double> > values;
};

#endif
