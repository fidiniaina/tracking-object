#include < stdio.h>
#include < iostream>

#include < opencv2\opencv.hpp>
#include < opencv2/core/core.hpp>
#include < opencv2/highgui/highgui.hpp>
#include < opencv2/video/background_segm.hpp>


#ifdef _DEBUG        
#pragma comment(lib, "opencv_core247d.lib")
#pragma comment(lib, "opencv_imgproc247d.lib")   //MAT processing
#pragma comment(lib, "opencv_objdetect247d.lib") //HOGDescriptor
//#pragma comment(lib, "opencv_gpu247d.lib")
//#pragma comment(lib, "opencv_features2d247d.lib")
#pragma comment(lib, "opencv_highgui247d.lib")
#pragma comment(lib, "opencv_ml247d.lib")
//#pragma comment(lib, "opencv_stitching247d.lib");
//#pragma comment(lib, "opencv_nonfree247d.lib");
#pragma comment(lib, "opencv_video247d.lib")
#else
#pragma comment(lib, "opencv_core247.lib")
#pragma comment(lib, "opencv_imgproc247.lib")
#pragma comment(lib, "opencv_objdetect247.lib")
//#pragma comment(lib, "opencv_gpu247.lib")
//#pragma comment(lib, "opencv_features2d247.lib")
#pragma comment(lib, "opencv_highgui247.lib")
#pragma comment(lib, "opencv_ml247.lib")
//#pragma comment(lib, "opencv_stitching247.lib");
//#pragma comment(lib, "opencv_nonfree247.lib");
#pragma comment(lib, "opencv_video247d.lib")
#endif 

using namespace cv;
using namespace std;



int main()
{

 //global variables
 Mat frame; //current frame
 Mat resizeF;
 Mat fgMaskMOG; //fg mask generated by MOG method
 Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
 Mat fgMaskGMG; //fg mask fg mask generated by MOG2 method
 

 Ptr< BackgroundSubtractor> pMOG; //MOG Background subtractor
 Ptr< BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
 Ptr< BackgroundSubtractorGMG> pGMG; //MOG2 Background subtractor
 


 pMOG = new BackgroundSubtractorMOG();
 pMOG2 = new BackgroundSubtractorMOG2();
 pGMG = new BackgroundSubtractorGMG();
 

 char fileName[100] = "C:\\POSCO\\video\\/cctv 2.mov"; //Gate1_175_p1.avi"; //mm2.avi"; //";//_p1.avi";
 VideoCapture stream1(fileName);   //0 is the id of video device.0 if you have only one camera   

 Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1,1) );   

 //unconditional loop   
 while (true) {   
  Mat cameraFrame;   
  if(!(stream1.read(frame))) //get one frame form video   
   break;
  
  resize(frame, resizeF, Size(frame.size().width/4, frame.size().height/4) );
  pMOG->operator()(resizeF, fgMaskMOG);
  pMOG2->operator()(resizeF, fgMaskMOG2);
  pGMG->operator()(resizeF, fgMaskGMG);
  //morphologyEx(fgMaskGMG, fgMaskGMG, CV_MOP_OPEN, element); 

 


  imshow("Origin", resizeF);
  imshow("MOG", fgMaskMOG);
  imshow("MOG2", fgMaskMOG2);
  imshow("GMG", fgMaskGMG);
  

  if (waitKey(30) >= 0)   
   break;   
 }

}


