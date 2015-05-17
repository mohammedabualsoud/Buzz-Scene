#include "build_dictionary.h"


int buildDictionary (const string& unclustered_descriptors_path ,size_t number_of_clusters , const string& directory_path ){

  FileStorage fsread (unclustered_descriptors_path , FileStorage::READ );
  if (!fsread.isOpened() ) {
    cerr << "Cannot opened file "<<unclustered_descriptors_path << "for reading " <<  endl ;
    return -2 ;
  }
  Mat unclustered_descriptors;
  Mat dicitnary ;
  fsread["Training-Images"] >> unclustered_descriptors ;
  fsread.release();

  cout << "Uncluster descriptors = " << unclustered_descriptors.rows << endl  ;

  BOWKMeansTrainer bowtrainer(number_of_clusters) ;
  bowtrainer.add(unclustered_descriptors ) ;
  cout << " Training Bovw Start .... " << endl ;
  dicitnary = bowtrainer.cluster () ;
  cout << " Training Finish " << endl ;
  cout << "Dicitnary rows = " << dicitnary.rows << endl ;

  stringstream sstm ;
  sstm << "dictionary-"  << number_of_clusters <<".yaml.gz";
  string dicitnary_filename = sstm.str() ;
  FileStorage fswrite(directory_path + dicitnary_filename , FileStorage::WRITE) ;
  if (!fswrite.isOpened() ) {
    cerr << "Cannot opened file for writing " << endl ;
    return -2 ;
  }
  cout << "Writing ... " << dicitnary.rows <<" Vocabulary Descriptors to " <<directory_path + dicitnary_filename << endl ;
  fswrite << "vocabulary"<<dicitnary ;
  fswrite.release();
  cout << "Writing operation done " << endl ;

  return 0 ;

}


int buildDictionary(const Mat& unclustered_desctriptors,Mat& dictinary , size_t number_of_clusters , const string& vocabulary_path){




  cout << "Uncluster descriptors = " << unclustered_desctriptors.rows << endl  ;

  BOWKMeansTrainer bowtrainer(number_of_clusters) ;
  bowtrainer.add(unclustered_desctriptors ) ;
  cout << " Training Bovw Start .... " << endl ;
  cout << " ........." << endl;
  dictinary = bowtrainer.cluster () ;
  cout << " Training Finish " << endl ;
  cout << "Dicitnary rows = " << dictinary.rows << endl ;

  stringstream sstm ;
  sstm << "dictionary-"  << number_of_clusters <<".yaml.gz";
  string dicitnary_filename = sstm.str() ;
  FileStorage fswrite(vocabulary_path + dicitnary_filename , FileStorage::WRITE) ;
  if (!fswrite.isOpened() ) {
    cerr << "Cannot opened file for writing " << endl ;
    return -2 ;
  }
  cout << "Writing ... " << dictinary.rows <<" Vocabulary Descriptors to " <<vocabulary_path+ "/" + dicitnary_filename << endl ;
  fswrite << "vocabulary"<<dictinary ;
  fswrite.release();
  cout << "Writing operation done " << endl ;

  return 0 ;

}

 
