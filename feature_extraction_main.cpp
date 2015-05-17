#include "feature_extraction.h"



/**
 * input : arguments count , arguments values 
 * output : depends on the function called from above (extractFeuature)
 * Functinaliy : call the extractFeuatre() function 
 **/
 

int main (int argc, char** argv){
  if ( argc < 3 ){
    cerr << "Usage : " << argv[0] << " <training_directory> <destination_directory> " << endl ; 
    return -2 ;
  }
  initModule_nonfree();
  //create SIFT detect  & descriptor 
  Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
  //Extract the whole features from 984 image and store them on disk.
  extractFeauture( detector ,  extractor , argv[1] , argv[2] ) ; 
  return 0 ;
}
