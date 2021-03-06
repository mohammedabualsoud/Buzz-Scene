
#include "train_svm.h"

/**
 * input :
 * argv : path to the histogram examples .
 * output : store the SVM parameters in <SVM_parameter_files> directory .
 *functionality : given the examples , a 24 SVM's will be train as one VS rest appoach ,
 * and it's corresponding parameters will be store in disk . 
 **/

int main (int argc , char** argv  ){

  if( argc < 2 ){
    cerr << "USAGE:: " << argv[0] << " <train examples file.yaml>"  << endl ;
    return -2 ;
  }
  string samples_pathfile ;
  samples_pathfile.assign ( argv[1] ) ;

  cout << "load Samples from disk.."<<endl;
  map<string,Mat> classes_training_data;
  FileStorage fs(samples_pathfile,FileStorage::READ);
  string prefix ="class_";
  //size_t num_samples  = 0 ;
  for (size_t i = 1 ; i <= 24; i++ ) {
    stringstream ss ;
    ss << prefix <<i ;
    string class_ = ss.str();
    //cout << class_<< endl ;
    fs[class_ ] >> classes_training_data[class_];
    //num_samples += classes_training_data[class_].rows  ;
    //cout << "Number of Samples from " << class_ << "= "<<classes_training_data[class_].rows  << endl ;
    
  }
  fs.release();
//    cout <<" Number of clasess  = " << classes_training_data.size()  << endl ;
//    cout << "Total Number of Samples  = " << num_samples << endl ;
  
  cout << "train SVM..as 1 vs 24 " <<endl;
  Mat& one_class = (*(classes_training_data.begin())).second;
  //cout << (*(classes_training_data.begin())).first;
  //cout << "Number of cols should be 800" << one_class.cols << "TYPE CV32F" <<  one_class.type() << endl ;
  trainSVM(classes_training_data, one_class.cols, one_class.type());
  cout << " Training SVM DONE " << endl ;

  
  return 0;
}
