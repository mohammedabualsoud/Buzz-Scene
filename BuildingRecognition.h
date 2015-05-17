#ifndef BUILDINGRECOGNITION_H
#define BUILDINGRECOGNITION_H

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/legacy/legacy.hpp"

using namespace std ;
using namespace cv ;

 //Requiremnets objecs
    //detecotr for keypoints (sift)
    //descriptor for features(sift)
    //vocabulary (dicitnary)
    //matcher of features
    //BOWdescriptorextractor
    //classifiers
    //vecotr of scores
    //the input image
/**
 * The wrapper class for the prediction .
 **/

class BuildingRecognition
{
 public:
  /**
   * Funcionality : initlize the whole members  . (SVMS classifier , Vocabulary , detector ,etc )
   **/

  BuildingRecognition () ;
  /**
   * input : 
   * input_img : the input image the wrapper should predict what class it belong to .
   * output_classes : the set candidate  classes for the <input_img> .
   * with_multiple_class : flag to enable multiple building recognition . 
   * output : the set of candidate classes . 
   * Functionality : the whole members will be initlized , then categorize_image(...) method is proceeded .
   **/

  BuildingRecognition(Mat& input_img , vector<string>& output_classes ,bool with_multiple_class = false,bool debug = false );
  /**
   * input :
   * input_img :  the input image the wrapper should predict what class it belong to .
   * output_classes : the set candidate  classes for the <input_img> .
   * output : the set of candidate classes . 
   * Functionality : the <input_image> will be quantized into 1000 dimention vector by the bowide member ,then the one vs rest (1 vs 24)
   * classification will proceed ,the class with heighest score win .
   **/
  
  void categorize_image(Mat& input_img , vector<string>& output_classes ) ;

  /**
   * input :
   * input_img :  the input image the wrapper should predict what class it belong to .
   * output_classes : the set candidate  classes for the <input_img> .
   * Functionality : this method based on pyrmid.
   * it will divide the image into 4 patches, then compute the responses for each,
   * the majority score will be selected as the best class .
   * if with_multiple_class flag is set ,then the....
   * (I think the accuracy depends on the number of patches taken from the test image
   * as the number of patches increased the accuracy also increased .
   * i will try later to make more patches.
   **/

  void categorize_image(Mat& input_img , map<string,vector<float> >& output_classes ) ;

  ~BuildingRecognition();

  void setWithMultiple_class();

  void setDebug();

 private:
 
  /**
   * Functionality : Initlize the SVM's paramerts from disk 
   **/
     void initSVMs();

     /**
      * Functionality : Initlize the Bag of visual words with the dictinary 
     **/
     void initVocabulary();
     

     
     Ptr<FeatureDetector > detector;
     Ptr<BOWImgDescriptorExtractor > bowide;
     Ptr<DescriptorMatcher > matcher;
     Ptr<DescriptorExtractor > extractor;
     map<string,CvSVM> classes_classifiers;
     Mat input_img;
     Mat vocabulary;
     bool with_multiple_class;//flag that enable multiple building recognition in the scene of the input image 
     bool debug;
     
};

#endif // BUILDINGRECOGNITION_H
