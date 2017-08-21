cmake_minimum_required(VERSION 2.8)

#project name 
PROJECT(objLeftNew) 

#Find OpenCV, you may need to set OpenCV_DIR variable
#to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)
#print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "	version: ${OpenCV_VERSION}")
message(STATUS "	libraries: ${OpenCV_LIBS}")
message(STATUS "	include path: ${OpenCV_INCLUDE_DIRS}")

link_directories(${MYPROJECT_DIR})
find_library(RUNTIME_LIB rt /usr/lib /usr/local/lib NO_DEFAULT_PATH)
# declare the executable target built from your source
add_executable(objLeftNew test.cpp objLeftDetectcpp.cpp myGMM.cpp)

#head file path
#INCLUDE_DERECTORIES( include )

#source directory
#AUX_SOURCE_DIRECTORY( src DIR_SRCS)

#set environment variable

#add link library
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})
