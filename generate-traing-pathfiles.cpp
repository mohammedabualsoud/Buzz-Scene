#include <iostream>
#include <string.h>
#include <string>
#include <dirent.h>
#include <fstream>



using namespace std ;
const string directory_prefix= "/home/abu-gasiem/CB-Wrokspace/Nablus-building-shops-recognition-bovw-module/test/";
int main (int argc , char** argv){
  if (argc <2){
    cerr << "Usage:./generate-train-pathfiles  [outputfile.txt]" << endl ; 
    return -1 ; 
  }
  string outputfilename = argv[1];
  for ( int i = 1 ; i <=24; i++ ){
    
  
  string directory_name = to_string(i);
  //read each entry in the directory ,then append it to outputfile
  DIR* dirp ;
  dirent* dep;
  fstream fs ;
  fs.open (outputfilename.c_str(), std::fstream::in | std::fstream::out | std::fstream::app);
  if ( !fs.is_open() ){
    cerr << "Cannot open file" << outputfilename <<endl ; 
    return -1 ; 
  }
  string dir_path = directory_prefix + directory_name ; 
  dirp = opendir(dir_path.c_str() );
  if (dirp == NULL){
    cerr << "Cannot open " << directory_name  << endl ;
    return  -2;
  }
  
  while ((dep = readdir(dirp)) != NULL) {
    if ( (strcmp(dep->d_name ,".") == 0) ||(strcmp(dep->d_name ,"..") == 0) ){
      continue ;
    }
    //format of line is as follow 
    //(location of file) (space) (class number)
    string line = "test/" +directory_name +"/" +dep->d_name +" " +directory_name;
    //insertline to the outputfile.txt; 
    fs  <<line << endl;
    
    
  }

  closedir(dirp) ;
  fs.close();
  }
}


