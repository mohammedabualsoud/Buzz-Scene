#include "feature_extraction.h"

int  extractFeauture(Ptr<FeatureDetector>& detector , Ptr<DescriptorExtractor>& dextractor ,
                    const string& directory_path , const string& destination_dirctory_path ){
  DIR* dirp ;
  dirent* dep;
  FileStorage fs;
  Mat descriptors;
  vector <string> images_container;
 


  //fill the container with Image names ,I use the vecotor container to easy the parallesim .
  dirp = opendir(directory_path.c_str() );
  if (dirp == NULL){
    cerr << "Cannot open " << directory_path  << endl ;
    return  -2;
  }

   string image_name;
  while ((dep = readdir(dirp)) != NULL) {
    if ( (strcmp(dep->d_name ,".") == 0) ||(strcmp(dep->d_name ,"..") == 0) ){
      continue ;
    }
    image_name.assign( dep->d_name );
    images_container.push_back(image_name) ;
  }

          
  closedir(dirp) ;


  //detect and descripe the whole features from Images container (984 img) .
#pragma omp parallel for num_threads(4) schedule(dynamic)
  for ( size_t i = 0 ; i < images_container.size() ; i++ ){
    string  filepath = directory_path + "/" +images_container[i] ;
    Mat desc ;
    Mat img = imread(filepath  ,CV_LOAD_IMAGE_GRAYSCALE) ;
    vector <KeyPoint> keypoints;
    //        if ( !img.data ){
    //            cerr << "Cannot read image" << filepath << endl ;
    //            return -2 ;
    //        }
    cout << "Extract Features from img " << i+1 << "...."<<endl ;
    detector->detect(img, keypoints);
    dextractor->compute(img, keypoints, desc);
#pragma omp critical
    {
      descriptors.push_back(desc);

    }
  }

  //store the whole unclustered descriptors into the <destination_directory_path> as yaml file .
  string destination = destination_dirctory_path +"/descriptors-984img.yaml.gz" ;
  fs.open(destination, FileStorage::WRITE);
  if ( !fs.isOpened() ){
    cerr << "Cannot open Filestorage " << endl ;
    return -2 ;
  }
  cout << "Number of Desctriptors Extracted = " << descriptors.rows <<" each with dimintion = "
       << descriptors.cols << endl;
  cout << "Writing Descrtiptors to File ..." << endl;
  fs <<"Training-Images" << descriptors ;
  fs.release();
  cout << "Write has been sucssfully done " << endl ;

  return 0 ;
}



int extractFeauture(Ptr<FeatureDetector>& detector , Ptr<DescriptorExtractor>& dextractor ,
                    const string& training_dir , Mat& unclustered_desctriptors ){

  DIR* dirp ;
  dirent* dep;
  vector <string> images_container;
  string image_name;

  //fill the container with Image names ,I use the vecotor container to easy the parallesim .
  dirp = opendir(training_dir.c_str() );
  if (dirp == NULL){
    cerr << "Cannot open " << training_dir  << endl ;
    return  -2;
  }


  while ((dep = readdir(dirp)) != NULL) {
    if ( (strcmp(dep->d_name ,".") == 0) ||(strcmp(dep->d_name ,"..") == 0) ){
      continue ;
    }
    image_name.assign( dep->d_name );
    images_container.push_back(image_name) ;
  }

  closedir(dirp) ;

   

  //detect and descripe the whole features from Images container (984 img) .
#pragma omp parallel for num_threads(4) schedule(dynamic)
  for ( size_t i = 0 ; i < images_container.size() ; i++ ){
    string  filepath = training_dir + "/" +images_container[i] ;
    Mat desc ;
    Mat img = imread(filepath  ,CV_LOAD_IMAGE_GRAYSCALE) ;
    vector <KeyPoint> keypoints;
    //        if ( !img.data ){
    //            cerr << "Cannot read image" << filepath << endl ;
    //            return -2 ;
    //        }
    cout << "Extract Features from img " << i+1 << "...."<<endl ;
    detector->detect(img, keypoints);
    dextractor->compute(img, keypoints, desc);
#pragma omp critical
    {
      unclustered_desctriptors.push_back(desc);

    }
  }

  cout << "Number of Desctriptors Extracted = " << unclustered_desctriptors.rows <<" each with dimintion = "
       << unclustered_desctriptors.cols << endl;

  return 0 ;


}

