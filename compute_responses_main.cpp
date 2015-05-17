
#include "compute_responses.h"




int main(int argc , char** argv ){
  initModule_nonfree();	
  const string vocabulary_dir = "dictionary/";
  if (argc < 4 ) {
    cerr  << "USAGE: " <<argv[0] << " <vocabulary_file.yml> <examples file> <response directory>  " <<endl;
    return -1;
  }

  string examples_file , response_dir;
  examples_file.assign( argv[2] ) ;
  response_dir.assign ( argv[3] ) ;
  // fetch the vocabulary/dictnary from disk into the BOW object .
  Mat vocabulary;
  FileStorage fs(vocabulary_dir + argv[1] , FileStorage::READ);
  if (!fs.isOpened() ){
    cerr << " cannot open" << argv[1] << endl ;
    return -1 ;
  }
  fs["vocabulary"] >> vocabulary;
  fs.release();
  //inilize the dectector,descriptor,matcher,bow objects .
  Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
  //L2 : the Ecluidean distance 
  Ptr<DescriptorMatcher > matcher(new  BruteForceMatcher<L2<float> >());
  BOWImgDescriptorExtractor bowide(extractor,matcher);
  bowide.setVocabulary(vocabulary);
  
  cout << "BOW has Vocabulary Size = " << vocabulary.cols << endl ;

  //setup training data for classifiers
  map<string,Mat> classes_training_data;
  classes_training_data.clear();

  cout << "compute the set of histograms for the set of examples ..." << endl;
   compute_responses(detector, bowide, classes_training_data ,examples_file , response_dir );

  cout << "Number of Classes has been extracted " << classes_training_data.size() << " classes." <<endl;
  for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
    cout  << (*it).first << " has " << (*it).second.rows << " samples(examples) " <<endl;
  }

  //cout << "training  SVMs" << endl ;
  //	string postfix = argv[2];
  //	trainSVM(classes_training_data, directory_name, bowide.descriptorSize(), bowide.descriptorType());

  return 0;
  
}
