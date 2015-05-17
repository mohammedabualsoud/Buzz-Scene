#include "compute_responses.h"


void  compute_responses(Ptr<FeatureDetector>& detector, BOWImgDescriptorExtractor& bowide, map<string,Mat>& classes_training_data,
			      string& examples_file ,string& response_dir ) {

  int total_samples = 0;
  fstream fs ;
  vector<string> train_imgfilenames;
  fs.open(examples_file.c_str() ) ;
  if ( !fs.is_open() ){
    cerr << "Cannot open "<<examples_file <<" file " << endl ;
    return ;
  }
  //fetch the images names into a container .
    string line ;
    while ( getline (fs,line) )
      {
	train_imgfilenames.push_back(line);

      }
    cout << "Whole number of training data = " << train_imgfilenames.size() << endl ;
    // multi threads for extraction the respone_histograms of train img.
	#pragma omp parallel for schedule(dynamic,3) num_threads(4)
	for(size_t i=0;i<train_imgfilenames.size();i++) {
  //printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());

  vector<KeyPoint> keypoints;
  Mat response_hist;
  Mat img;
  string filepath;
  string tmp;
  string class_;
  //extract the filepath and class label from the file 
  stringstream ss(train_imgfilenames[i] );
  // the format of the line was string then space then other string then number represent the class label .
  // this first two string represent the image file name . 
  ss >> tmp;
  filepath = tmp ;
  filepath += " ";
  ss >>tmp ;
  filepath += tmp ;
  ss >> class_ ;
  class_ = "class_" + class_ ;
  if(class_.size() == 0) continue;
  //compute the response for image i 
  img = imread(filepath);
  //if ( !img.data ){ return ;}
  detector->detect(img,keypoints);
  bowide.compute(img, keypoints, response_hist);

  cout << ".";
  cout.flush();

#pragma omp critical
  {
   //not yet created...
      if(classes_training_data.count(class_) == 0) { 
          classes_training_data[class_].create(0,response_hist.cols,response_hist.type());
      }
      classes_training_data[class_].push_back(response_hist);
      total_samples++;
  }


}
	cout << endl;
	cout << "Total_samples = " << total_samples << endl;
	cout << "save to file.."<<endl;
	// save the responses to disk as .yaml file 
	{

        stringstream ss;
        ss << "examples_"  <<classes_training_data["class_1"].cols << ".yaml" ;
        string outputfile = ss.str() ;
        cout << "outputfile : " << outputfile << endl ;
	FileStorage fs(response_dir + outputfile, FileStorage::WRITE);
	//save samples for each class on disk
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
             cout << "save  " << (*it).first << endl;
             fs << (*it).first << (*it).second;
         }

        }
}


