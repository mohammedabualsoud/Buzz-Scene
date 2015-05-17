#ifndef TEST_ALLCLASSES_H_INCLUDED
#define TEST_ALLCLASSES_H_INCLUDED



#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/legacy/legacy.hpp"

using namespace std ;
using namespace cv ;


/**
 * input : 
 * argv[1] :test samples file  .
 * argv[2] : SVM paramerters directory .
 * output : confustion matrix .
 * Functionality : Testing the module with one vs rest strategy .
 **/

int main(int argc , char** argv );

/**
 * input : 
 * w : wights for each class ,as  the data-set are unbalanced .
 * output : container of wights for classes  . 
 * Functionality : inilize the wights for each class in the data-set . 
 **/

void init(map<string,float>& w);

#endif // TEST_ALLCLASSES_H_INCLUDED
