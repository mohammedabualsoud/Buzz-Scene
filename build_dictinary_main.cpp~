#include "feature_extraction.h"
#include "build_dictionary.h"



int main (int argc , char** argv ){
  if ( argc < 4 ){
    cerr << "Usage : " << argv[0] << "  <Training directory>  <destination dictinary directory>  <size of the dicinary> " << endl ;
    return -2 ;
  }
  initModule_nonfree();
  int clusters = atoi(argv[3] );
  Mat uncluster_features  , dictinary ; 
  
  //create SIFT detect  & descriptor 
  Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
  //Extract the whole features from 984 image and store them on disk. 
  string train_dir;
  train_dir.assign(argv[1] ) ;
  extractFeauture(detector , extractor , train_dir , uncluster_features ); 
  //  clustering start into clusters vectors of dimention 128 .
  buildDictionary (uncluster_features  ,dictinary  ,(size_t) clusters , argv[2]  );

  return 0 ; 
}
