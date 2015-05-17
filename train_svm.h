#ifndef TRAIN_SVM_H_INCLUDED
#define TRAIN_SVM_H_INCLUDED

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;
/**
 * input :
 * <classes_training_data> : training examples of each class .
 * <response_cols> : dimention of the response vector (number of vocabulary ).
 * <response_type> the data type of the response vector .
 * Functionality : Given the Training examples , a supervised learing process will accomplish using the 24 suppor vector machines
 * one vs rest apporch is follwed .
 **/
void trainSVM(map<string,Mat>& classes_training_data, int response_cols, int response_type);

#endif // TRAIN_SVM_H_INCLUDED

