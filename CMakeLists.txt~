cmake_minimum_required(VERSION 2.8)
project( Buzz_Scene )
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -fopenmp")
endif()
find_package( OpenCV REQUIRED )
add_executable( extract_features feature_extraction_main.cpp feature_extraction.cpp )
add_executable( build_dictinary build_dictinary_main.cpp builed_dictionary.cpp feature_extraction.cpp)
add_executable(compute_histogram_responses compute_responses_main.cpp compute_responses.cpp)
add_executable(train_svms train_svm_main.cpp train_svm.cpp)
add_executable(predict_one_class predict_one_class.cpp)
add_executable(test_allclasses test_allclasses.cpp)
add_executable(building_recognition main.cpp BuildingRecognition.cpp)




target_link_libraries( extract_features ${OpenCV_LIBS} )
target_link_libraries( build_dictinary ${OpenCV_LIBS} )
target_link_libraries( compute_histogram_responses  ${OpenCV_LIBS} )
target_link_libraries( train_svms  ${OpenCV_LIBS} )
target_link_libraries( predict_one_class ${OpenCV_LIBS} )
target_link_libraries( test_allclasses ${OpenCV_LIBS} )
target_link_libraries( building_recognition ${OpenCV_LIBS} )

