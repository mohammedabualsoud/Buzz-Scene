#include "predict_one_class.h"


int main(int argc , char** argv ){
    initModule_nonfree();	
  if (argc < 3 ){
    cerr << "USAGE:: " << argv[0] << " <dictinary file> <test examples file> " << endl ; 
    return -2;
  }
 
  const string directory = "dictionary/" ;  
  string dictionary_file = directory + argv[1] ; 
  // in case you want to test two classifier with different parameters ,you can skip this and just test with one classifier .
  CvSVM classifier1,classifier2;
  Mat vocabulary;
  FileStorage fs(dictionary_file, FileStorage::READ);
  if (!fs.isOpened() ){

    cerr  << " cannot open " << dictionary_file  << endl ;
    return  -2 ;
  }
  //fetch the vocabulary from disk .
  fs["vocabulary"] >> vocabulary;
  fs.release();
  //inilize the detector,descriptor,matcher,bowdie
  Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
  Ptr<DescriptorMatcher > matcher(new  BruteForceMatcher<L2<float> >());
  BOWImgDescriptorExtractor bowide(extractor,matcher);
  //add the vocabulary to bovw .
  bowide.setVocabulary(vocabulary);

  cout << "BOW has Vocabulary Size = " << vocabulary.rows << endl ;
  //as i said before if your want to test the svm with different parameters , but now i will test one classifier with my svm_paramters
  //for class1 in mydata-set.
  //svm_1000_auto
  //svm_1000_c10e10_opt
  //classifier.load("SVM_parameter_files/svm_1000_auto/SVM_classifier_class_1.yaml") ;
  classifier1.load("SVM_parameter_files/SVM_classifier_class_1.yaml") ;
  //    classifier2.load("SVM_parameter_files/svm_1000_auto/SVM_classifier_class_1.yaml") ;

  //fetch the 48 examples from disk into a container  .
  fstream fstr (argv[2]);
  vector<string>files;
  if ( !fstr.is_open() ){
    cerr << "cannot open " <<argv[2]  << endl;
    return -2;
  }
  vector <string >lines ;
  int k=0;
  string line ;
  while ( getline (fstr,line) && k < 48 ){
    
    lines.push_back(line);
    k++;
  }
  fstr.close();
  //extract the filepath and class from test file
  string filepath;
  string tmp;
  string class_;
for ( size_t i = 0 ; i < lines.size() ; i++ ){

    stringstream ss(lines[i] );
    ss >> tmp;
    filepath = tmp ;
    filepath += " ";
    ss >>tmp ;
    filepath += tmp ;
    ss >> class_ ;
    class_ = "class_" + class_ ;
    //    classes_count[class_]++;
    files.push_back(filepath) ;

 } 
//compute the response histogram for each example ,then let the SVM of class1 predict the class label .
 for ( size_t i = 0 ; i  < files.size() ; i++ ){
   Mat img = imread(files[i] , 0 ) ;
   vector<KeyPoint> keypoints;
   detector->detect(img, keypoints);
   Mat imgDescriptor;
   //resopnse to the vocabulary
   bowide.compute(img, keypoints, imgDescriptor);
   string predicted_label ;
   //svm_1000_c130_g4
   {
     // if res == 1 then it's postive for class1 else (0) it's negative class (not class1 ) . 
     float res = classifier1.predict(imgDescriptor);
     float distance = classifier1.predict (imgDescriptor, true ) ;
     predicted_label = (res == 1 ? "Class 1 " : "NOT CLASS 1 ");
     std::cout << "- Result of prediction svm_1000_c130_g4 : (" << predicted_label << "): "
	       << "Label = " <<res <<"Distance: " << distance << std::endl;
   }
    // //this is another paramerts svm_1000_auto
    // {
    // float res = classifier2.predict(imgDescriptor);
    // float distance = classifier2.predict(imgDescriptor,true);

    //  predicted_label = (res == 1 ? "Class 1 " : "NOT CLASS 1 ");
    // std::cout << "- Result of prediction svm_1000_auto : (" << predicted_label << "): "
   //<< "Label = " << res <<"Distance: " << distance << std::endl;
    // }
   cv::imshow(predicted_label, img);
   cv::waitKey(-1);

   cv::destroyWindow(predicted_label);

 }
 //this is another testing if you want to check it, just some Scribble ... 

//    test_samples = imread("2015-04-17 17.46.39.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//    vector<KeyPoint> keypoints;
//    detector->detect(test_samples, keypoints);
//    Mat imgDescriptor;
//    //resopnse to the vocabulary
//    bowide.compute(test_samples, keypoints, imgDescriptor);
//    cout<< imgDescriptor.rows << endl ;
//
//    cout <<imgDescriptor.cols << endl;

 //testing class1 vs the negative examples of class_12
 //    fs["class_12"] >> test_sampels
 //    for ( int i = 0 ; i < test_samples.rows ; i++ ){
 //        float res = classifier.predict (test_samples.row(i) ) ;
 //        if (res == 1 )pos++ ;
 //        else neg++ ;
 //        }
 //        cout << "POS = " << pos << " ,  NEG = " << neg << endl ;

 //test svm classifier that classify class_1 as positive and others as negative
 //    string class_= "class_";
 //    for ( size_t i = 1 ; i <= 24; i++ ){
 //        stringstream ss ;
 //        ss << class_ <<i ;
 //        fs[ss.str()] >>test_samples;
 //        size_t positive  =  0 ;
 //        size_t negative = 0 ;
 //       
 //        for ( int i = 0 ; i < test_samples.rows ; i++ ){
 //            float res = classifier.predict(test_samples.row(i),false  ) ;
 //           ( (res == 1) ? (positive++):(negative++) );
 //
 //        }
 //
 //        cout << ss.str() << " positive examples  = " <<positive <<" , negative examples =" << negative  << endl ;
 //
 //    }
 //    fs.release();

}
