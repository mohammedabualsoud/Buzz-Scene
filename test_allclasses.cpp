#include "test_allclasses.h"


const string test_path_file = "test.txt";
const float TOTAL_TEST_IMG = 431;

int main(int argc , char** argv ){
  cout << "Testing Start ....." << endl;
  if ( argc < 3 ){
    cerr << "USAGE:: " << argv[0] << " <test samples file> <SVM parameters directory> " << endl ;
    return -2 ; 
  }
  fstream fs ;
  // confusionMatrix[classA][classB] = number of time B voted for A 
  map<string,map<string,int> > confusion_matrix;
  map<string,CvSVM*> classes_classifiers1;

  //the following commented code can be used if you don't store your examples on disk .


  //map<string,CvSVM*> classes_classifiers2;
  //vector <string> lines;
  //vector<string> files; //load up with images
  //vector<string> classes; //load up with the respective classes
  //map <string,int> classes_count;
  //fs.open(test_path_file.c_str() );
  //if ( !fs.is_open() ){
  //    cerr << "cannot open file" << test_path_file << endl ;
  //    return -1;
  //}
  //string line ;
  //while ( getline (fs,line) ){
  //
  //    lines.push_back(line);
  //}
  //fs.close() ;
  ////extract the filepath and class label from test.txt
  //string filepath;
  //string tmp;
  //string class_;
  //for ( size_t i = 0 ; i < lines.size() ; i++ ){
  //
  //    stringstream ss(lines[i] );
  //    ss >> tmp;
  //    filepath = tmp ;
  //    filepath += " ";
  //    ss >>tmp ;
  //    filepath += tmp ;
  //    ss >> class_ ;
  //    class_ = "class_" + class_ ;
  //    classes_count[class_]++;
  //    files.push_back(filepath) ;
  //    classes.push_back(class_);
  //
  //}
  //for (map<string,int>::iterator it = classes_count.begin(); it != classes_count.end(); ++it) {
  //    cout << it->first <<" : " <<  it->second << endl ;
  //    }

  //declare the decscriptor & load the vocabulary into bowdescriptor
  //    Mat vocabulary;
  //	FileStorage fs1("dictionary/dictionary-800.yaml", FileStorage::READ);
  //    if (!fs1.isOpened() ){
  //
  //        cout << " cannot open" << "dicitnary file" << endl ;
  //        return -1  ;
  //    }
  //	fs1["vocabulary"] >> vocabulary;
  //	fs1.release();
  //
  //	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
  //    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
  //	Ptr<DescriptorMatcher > matcher(new  BruteForceMatcher<L2<float> >());
  //	BOWImgDescriptorExtractor bowide(extractor,matcher);
  //	bowide.setVocabulary(vocabulary);
  //
  //	cout << "BOW has Vocabulary Size = " << vocabulary.rows << endl ;

  //load the classifier parameters from disk .
  {
    
    string params_prefix = "SVM_parameter_files/";
    params_prefix.append ( argv[ 2 ] ) ;
    params_prefix.append ("/SVM_classifier_class_"  );
    
    const string class_ = "class_";
    for ( size_t i = 1 ; i <=24; i++ ){
      stringstream ss1 , ss2 ;
      ss1 << class_  << i ;
      ss2 << params_prefix << i  << ".yaml";
      string label = ss1.str() ;
      string params_path = ss2.str() ;
      
      classes_classifiers1[label] = new CvSVM ();
      classes_classifiers1[label]->load(params_path.c_str() ) ;
      
    }
  }

  map <string , Mat> test_examples;
  string test_samples = "test_samples/";
  test_samples.append( argv[1] ) ; 
  FileStorage fs2 (test_samples , FileStorage::READ) ;
  if (!fs2.isOpened() ){
    
    cerr << " cannot open" << argv[1] << " test examples "  << endl ;
    return -2  ;
  }
  //read the test examples from disk,into test_examples container .
  {
    string class_ = "class_" ;
    for ( size_t i = 1 ; i <= 24 ; i++ ){
      stringstream ss;
      ss <<class_ << i ;
      fs2[ss.str().c_str()] >> test_examples[ss.str()] ;
      // cout <<test_examples[ss.str()].rows;
    }


  }
fs2.release();
    // extract/compute the responses of  test files .
//{

    //for(size_t i = 0 ; i < files.size() ; i++ ) {
    //   Mat img = imread(files[i],0),resposne_hist;
    //   vector<KeyPoint> keypoints;
    //   detector->detect(img,keypoints);
    //   bowide.compute(img, keypoints, resposne_hist);
    //   //cout << "row x cols =" << resposne_hist.rows <<"x" << resposne_hist.cols << endl ; //1k
    //
    //
    //   response_examples[ classes[i] ].push_back(resposne_hist ) ;
//}


//inilize the confusion matrix
{
  for (map<string,CvSVM*>::iterator it = classes_classifiers1.begin(); it != classes_classifiers1.end(); ++it) {
    string class1 = it->first ;
    for (map<string,CvSVM*>::iterator it1 = classes_classifiers1.begin(); it1 != classes_classifiers1.end(); ++it1) {
      string class2 = it1->first ;
      confusion_matrix[class1][class2] = 0;
    }
  }
}

// 1vs rest classification
// for each examples in the container ,the classifier with  The most negative  score computed   will be the predicted class .
 for (map<string , Mat>::iterator it1 = test_examples.begin(); it1 != test_examples.end(); ++it1) {
   Mat class_examples = it1->second;
   for (int i = 0; i < class_examples.rows; i++ ){
     string current_class = it1->first ;
     
     float minf = FLT_MAX;
     string minclass;
     //test with all classifier .
     for (map<string,CvSVM*>::iterator it = classes_classifiers1.begin(); it != classes_classifiers1.end(); ++it) {
       float res = (*it).second->predict(class_examples.row(i),true);
       if (res < minf) {
	 minf = res;
	 minclass = (*it).first;
       }

     }
     confusion_matrix[current_class][minclass]++;
   }

 }
 //the confusion matrix .
 Mat confus (24,24,CV_8U,Scalar (0));
 //copy the confusion_matrix map into a Matrix object ,as the Mat object is easier to access it's elements by numbers,
 //also the elements need to be sorted in accending order .
 {

   for ( int i = 1 ; i <= 24; i++ ){
     string class1;
     stringstream ss1 ;
     ss1 << "class_" << i ;
     class1 = ss1.str() ;
     
     for (int j = 1; j <= 24; j++ ){
       string class2;
       stringstream ss2 ;
       ss2 << "class_" << j ;
       class2 = ss2.str() ;
       confus.at<uchar>(i-1,j-1) = confusion_matrix[class1][class2];
     }
	
   }
 }
 //print the confusion matrix in python format ,this format let you visualize the confustion matix by another python program.
 //you can redierct the output to the disk to best view .
 cout << format(confus,"python");
 //compute the whole accuracy ,NOTE: this is not the correct  performace measurement , because my data-set not balanced . 
 {
   //as you can see the diagnal represnt the True positive rate of the Module . 
   float true_pos =0  ;
   float num_of_samples = 431 ;
   int i =0;
   int j = 0 ;
   while ( i < 24 && j < 24 ){
     true_pos += confus.at<uchar> (i,j) ;
     i++;
     j++;
   }
   cout << endl
        << "Accuracy = "  << ( (true_pos/num_of_samples)*100 ) << endl ;

 }
 //compute the precion ,and recall for each classifier,with wights .
 //this one is the standard way of measureing the performace of the Module when dealing with unbalanced data-set .
 //Note this is an iterative computation,each iteration will compute a classifier precion & recall  .
 {

   //precision of the  classes
   map<string ,float> precisions;
   //recall of the classes
   map<string,float> recall;
   //wights of  the classes
   map <string , float> wights ;
   //inilize the wights for each class .
   init(wights);
   //the diagnoal entry in the confusion matrix for a class (i) = mat(i,i)
   vector<size_t> true_pos;
   //the sum  column of confusion matrix for a class(i) without the entry (i,i)
   vector<size_t> false_pos;
   //the sum of row (i) without row(i)cols(i) (i.e without the true pos)
   vector<size_t> false_neg;

   for ( size_t col = 0 ; col < 24 ; col++ ){
     size_t tp = confus.at<uchar>(col,col) ;
     size_t false_positive =0.0;
     size_t false_negative =0.0;
     true_pos.push_back(tp) ;
     //sumation of all column for classifier i (false positive) with out the true_pos
     for ( size_t row = 0 ; row < 24 ; row++ ){
       //ignore the true positive entry .
       if ( col == row )
	 continue ;
       false_positive +=  confus.at<uchar>(row,col) ;
     }
     false_pos.push_back(false_positive) ;
     // the precision for class(col)  i.e class1 ... class24
        float precision = ( (float)tp/(tp + false_positive ) ) * 100;
        stringstream ss ;
        ss <<"class_";
        ss <<  col+1 ;
        precisions[ss.str() ] = precision ;
        //compute the false negative for each class
        //now col index is represnt row i ,and k represent col k
        for ( size_t k = 0 ; k < 24; k++ ){
            //this entry is t.p , Ignore it 
            if (col == k ){
                continue ;
            }
            false_negative += confus.at<uchar>(col , k);
        }
	//the recall for class(col) i.e class1 ... class24
        float recall_i =  ( (float)tp / (tp + false_negative ) )*100 ;
        recall[ss.str()] = recall_i  ;

    }
    //display the precions,compute the average prection & recall for the Whole Module.
   float avg_precion = 0.0;
   for (map<string,float>::iterator it = precisions.begin(); it != precisions.end(); ++it) {
     cout << "Precion of " << it->first << " = " <<it->second << "%" << endl ;
     avg_precion += it->second * wights[it->first];
   }
   cout << endl << endl ;
   //display the recall
   float avg_recall = 0.0;
   for (map<string,float>::iterator it = recall.begin(); it != recall.end(); ++it) {
     cout << "Recall of " << it->first << " = " <<it->second << "%" << endl ;
     avg_recall += it->second * wights[it->first]  ;
   }
   cout << endl << endl ;

   cout << "Average Precsion = " << (avg_precion) <<"%" << endl ;
   cout << "Average Recall = " << (avg_recall) << "%" << endl ;
 }


 //release the resources 
 for (map<string,CvSVM*>::iterator it = classes_classifiers1.begin(); it != classes_classifiers1.end(); ++it) {
   delete it->second ;
 }
//    //store the response examples on disk
//    string filename ;
//    stringstream ss;
//    ss << "test_samples/test_examples_" <<response_examples["class_1"].cols <<".yaml" ;
//    filename = ss.str() ;
//    FileStorage response_file(filename.c_str() , FileStorage::WRITE) ;
//    if (!response_file.isOpened() ){
//
//        cout << " cannot open  "  << filename <<"for writing "  << endl ;
//        return -1  ;
//    }
//    for (map<string,Mat >::iterator it = response_examples.begin(); it != response_examples.end(); ++it){
//        response_file << it->first << it->second ;
//    }
//    cout <<"Write the test response  examples to vocabulary of size " << response_examples["class_1"].cols
//        << "is done succfully " << endl ;
//    response_file.release() ;

return 0;
}


void init(map<string,float>& w){
    w["class_1"] = 48/TOTAL_TEST_IMG;
    w["class_2"] = 12/TOTAL_TEST_IMG;
    w["class_3"] = 14/TOTAL_TEST_IMG;
    w["class_4"] = 27/TOTAL_TEST_IMG;
    w["class_5"] = 10/TOTAL_TEST_IMG;
    w["class_6"] = 14/TOTAL_TEST_IMG;
    w["class_7"] = 9/TOTAL_TEST_IMG;
    w["class_8"] =  6/TOTAL_TEST_IMG;
    w["class_9"] =  27/TOTAL_TEST_IMG;
    w["class_10"] =  10/TOTAL_TEST_IMG;
    w["class_11"] =  4/TOTAL_TEST_IMG;
    w["class_12"] =  28/TOTAL_TEST_IMG;
    w["class_13"] =  5/TOTAL_TEST_IMG;
    w["class_14"] =  11/TOTAL_TEST_IMG;
    w["class_15"] =  23/TOTAL_TEST_IMG;
    w["class_16"] =  27/TOTAL_TEST_IMG;
    w["class_17"] =  12/TOTAL_TEST_IMG;
    w["class_18"] =  28/TOTAL_TEST_IMG;
    w["class_19"] =  14/TOTAL_TEST_IMG;
    w["class_20"] =  8/TOTAL_TEST_IMG;
    w["class_21"] =  6/TOTAL_TEST_IMG;
    w["class_22"] =  20/TOTAL_TEST_IMG;
    w["class_23"] =  24/TOTAL_TEST_IMG;
    w["class_24"] =  44/TOTAL_TEST_IMG;
}
