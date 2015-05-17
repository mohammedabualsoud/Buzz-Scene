
#ifndef FEATURE_EXTRACTION_H_INCLUDED
#define FEATURE_EXTRACTION_H_INCLUDED



#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>






using namespace cv;
using namespace std ;


/**
 * input : detector , descroptor  , traing_directory, destination_directory
 * output : store the descriptors on disk.
 * Functionality : 
 * Detect and descripe the Features from whole Images within directory <training_dir> (984 Image ) and store them in .yaml file
 * within <destination_directory> directory .
 **/

int extractFeauture(Ptr<FeatureDetector>& detector , Ptr<DescriptorExtractor>& dextractor ,
                    const string& training_dir , const string& destination_dirctory );


/**
 * input : detector , descroptor  , traing_directory 
 * output : Matrix object container 
 * Functionality : 
 * Detect and descripe the Features from whole Images within directory <training_dir> (984 Image ) and store them in the 
 * Container <uncluster_descriptors> 
 **/

int extractFeauture(Ptr<FeatureDetector>& detector , Ptr<DescriptorExtractor>& dextractor ,const string& training_dir ,Mat& uncluster_descriptors);
 
#endif // FEATURE_EXTRACTION_H_INCLUDED
