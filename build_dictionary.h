#ifndef BUILD_DICTIONARY_H_INCLUDED
#define BUILD_DICTIONARY_H_INCLUDED




#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


/**
 * input : <unclustered_descriptors_path> : uncluster words/descriptors directory , <number_of_clusters> : number of centers will be created
 * , <dictinary_dir> :  directory to store the created dictinary .
 * output : dictinary with <number_of_cluters> is stored on disk in <dictinary_dir> . 
 * functionalty : This is very long computaion on my dataset 984 , it's a Quantization process (unsupervied learning)
 * using the kmean++ algorithem. given the examples (the descriptors in <unclustered_descriptors_path> ) and number of center ,will end up
 * with the Dictinary of size <number_of_clusters> .
 **/

int buildDictionary(const string& unclustered_descriptors_path , size_t number_of_clusters , const string& dictinary_dir );

/**
 * this method same the above one ,but the unclutered descriptors are read from a Matrix object not from a file on disk . 
 **/

int buildDictionary(const Mat& unclustered_desctriptors, Mat& dictinary ,size_t number_of_clusters , const string& vocabulary_path);




#endif // BUILD_DICTIONARY_H_INCLUDED
