
#include "train_svm.h"

//CvParamGrid Cgrid=CvSVM::get_default_grid(CvSVM::C);
//CvParamGrid gammaGrid=CvSVM::get_default_grid(CvSVM::GAMMA);
//CvParamGrid pGrid=CvSVM::get_default_grid(CvSVM::P);
//CvParamGrid nuGrid=CvSVM::get_default_grid(CvSVM::NU);
//CvParamGrid coeffGrid=CvSVM::get_default_grid(CvSVM::COEF);
//CvParamGrid degreeGrid=CvSVM::get_default_grid(CvSVM::DEGREE);






void trainSVM(map<string,Mat>& classes_training_data, int response_cols, int response_type) {


  //fetch the classes labels into a container.
  vector<string> classes_names;
  for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
    classes_names.push_back((*it).first);
  }
  int class_successfully_trained = 0 ;
  //Train one vs rest SVMs (1 vs 24) 
  //Each SVM will feeded by same examples , but each one will take different different positive label ,and negative labels .
#pragma omp parallel for schedule(dynamic) num_threads(4)
  for (size_t i=0; i<classes_names.size() ;i++) {
    string class_ = classes_names[i];
    cout << "Thread " << omp_get_thread_num() << " training class: " << class_ << ".." << endl;

    //container matrix contains copy of all examples .
    Mat samples(0,response_cols,response_type);
    //container matrix contains label for each example . 
    Mat labels(0,1,CV_32FC1);

    //copy class samples and label
    //cout <<"TRAIN("<< class_ <<")::" << "adding " << classes_training_data[class_].rows << " positive samples" << endl;
    //copy the positive examples and labels 
    samples.push_back(classes_training_data[class_]);
    Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
    labels.push_back(class_label);

    //copy rest negative examples and labels
    for (map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
      string negative_class = (*it1).first;
      //ignore the postive examples 
      if(negative_class.compare(class_)==0) continue;
      samples.push_back(classes_training_data[negative_class]);
      class_label = Mat::zeros(classes_training_data[negative_class].rows, 1, CV_32FC1);
      labels.push_back(class_label);
      //	cout << "TRAIN(" << class_ << " )::" << "adding"
      //	<< classes_training_data[negative_class].rows  <<"of " << negative_class << "negative samples" << endl ;

    }
    //for now the examples are two type class_ is pos and negative_class is negative
    // feed the SVM with examples ,and start  training.
    cout << "Train.." << endl;
    // convert the samples type into 32float .
    Mat samples_32f;
    samples.convertTo(samples_32f, CV_32F);
    if(samples.rows == 0) {cout <<"NTH CLASS!" ; continue ; } //NTH class?!
    CvSVM classifier;
    CvSVMParams params;
    //best configuration
    params.C = 130;
    params.gamma =4;
    params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100000000, FLT_EPSILON );
    //feed the classifier i with samples & corresponding labeles , & paramerter.
    if (classifier.train(samples_32f,labels, Mat(), Mat(), params ) ) {

#pragma omp atomic
      class_successfully_trained++;

    }
    int num_support_vec     = classifier.get_support_vector_count();
    cout << "Number of Support vectors = " << num_support_vec << endl ;
    //store the SVM parameter for each class
    {
      stringstream ss;
      ss << "SVM_parameter_files/";
      ss << "SVM_classifier_";
      ss << class_ << ".yaml";

      cout << "Save.." << endl;
      classifier.save(ss.str().c_str());
    }
    cout << "Training " << class_ << "finish ,and it's parameter file stored on disk" <<endl ;
  }
  cout  << "Number of successfully trained classes = " << class_successfully_trained << endl ;
}

