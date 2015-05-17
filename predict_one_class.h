#ifndef PREDICT_ONE_CLASS_H_INCLUDED
#define PREDICT_ONE_CLASS_H_INCLUDED


#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace cv ;
/**
 * input : 
 * argv[1] : dictinary file to be used by the bag of visual words extractor . 
 * argv[2] : the test examples file .
 * output : 
 * set of predictions operation done by a specific SVM classifier in my case class 1 is used .
 * Functionality : an arbitrary class will be tested against set of postive & negative examples from the provided file test.txt
 * all examples will be quatized and feed to the SVM with it's corresponding label for prediction .
 **/

//int  main(int argc , char** argv );
#endif // PREDICT_ONE_CLASS_H_INCLUDED
