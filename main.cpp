
#include "BuildingRecognition.h"

/**
 * Functionality : maping  the classes labels into a real names . 
 **/
void init();
//map for mapping the classes labels .
map<string,string> classes_names;

/**
 * input :
 * argv[1] : unseen image .
 * output : predict the label of the image . 
 * Functionality : let the wrapper buildingRecoginion predict the Image label . 
 **/


int main(int argc , char** argv ){
  initModule_nonfree();
  if (argc < 2 ){
    cerr << "USAGE:: " << argv[0] << " <input_image>  " << endl ;
    return -1 ;
  }
  init();
  Mat test_img ;
  //wrapper for the predction operation.
  BuildingRecognition brecognition ;
  //vector contain the candidates classes .
  vector<string> responses ;
  test_img = imread(argv[1] , 0  );
  if (!test_img.data ){
    cerr << "Cannot read " << argv[1]  << endl ;
    return -1 ;
  }
  //the first class in the container responses is the winner class . 
  brecognition.categorize_image(test_img,responses) ;
  string win_class = classes_names[responses[0] ];
  cout << "Best Class : " <<  win_class<< endl ;
  //Display the building label on the input image .
  putText(test_img,win_class , Point( (test_img.rows/2), (test_img.cols/2)), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255,0,0), 2);
  imshow(win_class , test_img ) ;
  waitKey(-1);


  return 0;
}


void init(){

classes_names["class_1"] = "Al-Borj" ;
classes_names["class_2"] = "Crocs" ;
classes_names["class_3"] = "Setra";
classes_names["class_4"] = "Kena_ll_Feda" ;
classes_names["class_5"] = "Al-Safie";
classes_names["class_6"] = "Chicken-hut";
classes_names["class_7"] = "Al-Waha";
classes_names["class_8"] = "Masa";
classes_names["class_9"] = "Bon-al_Tareq";
classes_names["class_10"] = "Online";
classes_names["class_11"] = "Music-Hard-Rock";
classes_names["class_12"] = "Boquet";
classes_names["class_13"] = "Sys Com";
classes_names["class_14"] = "Nour";
classes_names["class_15"] = "Mr-Peker";
classes_names["class_16"] = "Home-EXPO";
classes_names["class_17"] = "Al-Sosa TM";
classes_names["class_18"] = "Bet-Altefel";
classes_names["class_19"] = "G-6";
classes_names["class_20"] = "Legent-Shop";
classes_names["class_21"] = "Bon-Bon";
classes_names["class_22"] = "Pizza-Atot";
classes_names["class_23"] = "Red";
classes_names["class_24"] = "Tabele TM";

}
