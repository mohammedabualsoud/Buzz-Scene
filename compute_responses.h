#ifndef COMPUTE_RESPONSES_H_INCLUDED

#define COMPUTE_RESPONSES_H_INCLUDED




#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <dirent.h>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/legacy/legacy.hpp>


#include <omp.h>


using namespace cv;
using namespace std;

/**
 * input : <detector> : detector ,<bowide> : Bag of visual world descriptor , <image_file>  : the examples file name 
 * <response_dir> the directory where to store the examples responses to the dictionary .
 * output : the responses of each examples within the <image_file> computed ,and stored in <response_dir> as a .yaml file .
 * Functionality : Given the detector , bowide , examples (images) , and the destination directory , then the responses
 * for the dicinary will be computed.
 **/

void compute_responses(Ptr<FeatureDetector>& detector, BOWImgDescriptorExtractor& bowide, map<string,Mat>& classes_training_data,
			      string& image_file ,string& response_dir ) ;


#endif // COMPUTE_RESPONSES_H_INCLUDED


