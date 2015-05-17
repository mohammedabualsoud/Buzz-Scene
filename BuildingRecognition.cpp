#include "BuildingRecognition.h"

BuildingRecognition::BuildingRecognition(){
  cout << "BuildingRecognition::Default constructer " << endl ;
  initSVMs();
  initVocabulary() ;
  Ptr<FeatureDetector> _detector = FeatureDetector::create("SIFT");
  Ptr<DescriptorExtractor> _extractor = DescriptorExtractor::create("SIFT");
  Ptr<DescriptorMatcher > _matcher(new  BruteForceMatcher<L2<float> >());
  matcher = _matcher;
  detector = _detector;
  extractor = _extractor;
  bowide = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractor,matcher));
  bowide->setVocabulary(vocabulary);
  with_multiple_class  = false ;
  debug = false ;
}

BuildingRecognition::BuildingRecognition (Mat& input_img , vector<string>& output_classes
					  ,bool with_multiple_class ,bool debug ):BuildingRecognition(){
  this->input_img =  input_img ;
  categorize_image(input_img , output_classes ) ;
  this->debug = debug ;
  this->with_multiple_class = with_multiple_class ;
}






void BuildingRecognition:: initSVMs(){
  if(debug){cout << "Inilizing the Classifers " << endl ; }
  const string params_prefix = "SVM_parameter_files/svm_1000_c130_g4/SVM_classifier_class_";
  const string class_label = "class_" ;
  string path ;
  for ( size_t i = 1 ; i <= 24 ; i++ ){
    stringstream ss ;
    ss << params_prefix << i  << ".yaml"  ;
    path = ss.str();
    ss.str(string());
    ss <<class_label << i ;

    classes_classifiers[ss.str() ].load(path.c_str() ) ;
  }
  if(debug){ cout << "SVMS parameters successfully loaded" << endl ; }
}


void BuildingRecognition:: initVocabulary(){
  if(debug){cout << "Inilizing the Vocabulary " << endl ; }
  const string vocabulary_path = "dictionary/dictionary-1000.yaml.gz" ;
  FileStorage fs (vocabulary_path , FileStorage::READ ) ;
  if ( !fs.isOpened () ){
    if(debug){ cout << "Cannot open " << vocabulary_path <<" to load Vocabulary" << endl ; }
    return ;
  }
  fs["vocabulary"] >> vocabulary ;
  fs.release() ;
  if(debug){ cout << "Vocabulary loaded ,it's vector size" << vocabulary.cols <<
      "number of visual words = " << vocabulary.rows << endl ; }

}

/*Evalute the input image to classifiy to class x */
void BuildingRecognition:: categorize_image(Mat& input_img , vector<string>& output_classes ) {
    if(debug){ cout << "Start the Categorization ..." << endl ;}
    this->input_img = input_img ;
    vector <KeyPoint> keypoints ;
    if (this->input_img.rows != 480 && this->input_img.cols != 640 ){
        if (debug){ cout << "Image Size error ,should be 640X480" << endl ; }
        return ;
    }
    Mat response_histogram ;
    detector->detect(this->input_img, keypoints ) ;
    bowide->compute ( this->input_img , keypoints , response_histogram )  ;
    if(debug ) { cout << "response histogram for the input Image :" << response_histogram.size() << endl ; }
    if ( response_histogram.rows == 0 ){
        return ;
    }
//    start the classification 1vs rest , the highest score wins

//    vector<string> classes ;
//    for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
//          classes.push_back(it->first) ;
//
//       }

    float minf = FLT_MAX;
    string minclass;
//    string class_ = "class_" ;
//    vector<string> min_c;
//    vector<float> min_score;
//    #pragma omp parallel num_threads(4)
//    {
//
//        float local_minf = FLT_MAX ;
//        string local_minclass;
//        #pragma omp  parallel for schedule(static,6)
//        for (size_t i = 0 ; i < classes.size() ; i++ ){
//            float res = classes_classifiers[classes [ i ] ].predict(response_histogram , true ) ;
//            if ( res < local_minf ){
//                local_minf = res  ;
//                local_minclass = classes[i];
//            }
//
//        }
//        #pragma omp critical
//        {
//            min_c.push_back(local_minclass);
//            min_score.push_back(local_minf );
//        }
//    }
//    for ( size_t i = 0 ; i < min_score.size() ; i++ ){
//        if ( min_score[i] < minf ){
//            minf = min_score[i] ;
//            minclass = min_c[i] ;
//        }
//    }
    //1vsRest classifiers
    clock_t begin = clock();
   // for ( size_t i = 0  ; i < 100; i++ ){
        for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {

                float res = (*it).second.predict(response_histogram,true);
                if (res < minf) {
                    minf = res;
                    minclass = (*it).first;
                }

           }
     // }
       clock_t end = clock();
       double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
       cout << "Classifyication Time = " << (elapsed_secs *1000)<< "msec" << endl ;

       output_classes.push_back(minclass) ;
//       putText(this->input_img, minclass, Point( (input_img.rows/2), (input_img.cols/2)), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255,0,0), 2);
//       imshow(minclass , this->input_img ) ;
//       waitKey(-1);
    return ;
}
/*
this method based on pyrmid.
it will divide the image into 4 patches, then compute the responses for each,
the majority score will be selected as the best class .
if with_multiple_class flag is set ,then the....
(I think the accuracy depends on the number of patches taken from the test image
as the number of patches increased the accuracy also increased .
i will try later to make more patches
*/
void BuildingRecognition:: categorize_image(Mat& input_img , map<string,vector<float> >& output_classes ) {
     if(debug){ cout << "Start the Categorization ..." << endl ;}
     vector<Mat> image_parts;
     this->input_img = input_img ;
    if (this->input_img.rows != 480 && this->input_img.cols != 640 ){
        if (debug){ cout << "Image Size error ,should be 640X480" << endl ; }
        return ;
    }

    Size img_size = input_img.size();
    size_t width = img_size.width;
    size_t height = img_size.height;
    //divide the image into 4 patches.
    //first row patches
    for (size_t x = 0, y = 0 ; x < width ; x += width/2 ){
        Rect temp (x , y , width/2 , height/2 ) ;
        image_parts.push_back (input_img(temp ).clone()) ;
    }
    //second row patches
     for (size_t x = 0, y = height/2 ; x < width ; x += width/2 ){
        Rect temp (x , y , width/2 , height/2 ) ;
         image_parts.push_back (input_img(temp).clone() ) ;
    }

    // compute the responses for  image parts  of the test_image.
    vector<Mat> response_histograms ;

    for (size_t i = 0 ; i < image_parts.size() ; i++ ){

        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detect( image_parts[i], keypoints ) ;
        bowide->compute(image_parts[i],keypoints,descriptor ) ;
        response_histograms.push_back(descriptor ) ;

        if(debug ) { cout << "response histogram for the input Image :" << response_histograms[i].size() << endl ; }
        if ( response_histograms[i].rows == 0 ){
            if(debug ){ cout << "no response computed  " << endl ; }
            return ;
        }

    }

    //start classification  1vs  rest classifier
    for ( size_t i = 0 ; i < response_histograms.size() ; i++ ){
         float minf = FLT_MAX;
         string minclass;
         for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
                float res = (*it).second.predict(response_histograms[i],true);
                if (res < minf) {
                    minf = res;
                    minclass = (*it).first;

                }
       }
      // store the scores
       output_classes[minclass].push_back(minf);
       if(debug){ cout << " min class " << i+1 << "is " << minclass << endl  ; }
    }
    if(debug){cout << "number of classes predicted =" << output_classes.size() ; }
    //just one class predict
    if (!with_multiple_class){
        size_t max_class_score  =0;
        string won_class;
        for (map<string,vector<float> >::iterator it = output_classes.begin(); it != output_classes.end(); ++it) {
            if (it->second.size() > max_class_score ){
                won_class = it->first;
                max_class_score = it->second.size() ;
            }
    }
    if (max_class_score >= 3 ){
        if(debug){ cout << "Max class is : " << won_class << endl ; }

    }else if  ( max_class_score == 1 ){
        //go iterate through the 4 classes and find the minMum score (tie between 4 classes)
        float min_class = FLT_MAX;
        for (map<string,vector<float> >::iterator it = output_classes.begin(); it != output_classes.end(); ++it) {
            if ( it->second[0] < min_class ){
                min_class = it->second[0] ;
                won_class = it->first ;
            }
        }
        if(debug){ cout << "Max class is : " << won_class << endl ; }
    }else if (max_class_score == 2 ){ //there's tie between 2 classes
        float sum_class1_score  = output_classes[won_class][0] +output_classes[won_class][1] ;
        float sum_class2_score = 0.0;
        string class2;
         for (map<string,vector<float> >::iterator it = output_classes.begin(); it != output_classes.end(); ++it) {
                if (it->first.compare(won_class) == 0 ){ continue ; }
                sum_class2_score = it->second[0] + it->second[1] ;
                class2 = it->first;
         }
         if ( sum_class1_score > sum_class2_score ){
            won_class = class2 ;
         }
         if(debug){ cout << "Max class is : " << won_class << endl ; }
    }

//    putText(this->input_img, won_class, Point( (input_img.rows/2), (input_img.cols/2)), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255,0,0), 2);
//    imshow(won_class , this->input_img ) ;
//    waitKey(-1);
//
//        imshow("a",image_parts[0] );
//        imshow("b",image_parts[1] );
//        imshow("c",image_parts[2] );
//        imshow("d",image_parts[3] );
//        waitKey(-1);


    }else if ( with_multiple_class ){
        //TO DO LATER
    }


  return ;
}
BuildingRecognition::~BuildingRecognition()
{
    //dtor
    if(debug){cout << "BuildingRecognition::Destruction Object" ; }
}
