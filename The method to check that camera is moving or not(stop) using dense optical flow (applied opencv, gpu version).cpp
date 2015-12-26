#include < stdio.h>

#include < opencv2\opencv.hpp>
#include < opencv2/core/core.hpp>
#include < opencv2/highgui/highgui.hpp>
#include < opencv2\gpu\gpu.hpp>
#include < opencv2\nonfree\features2d.hpp >    



#ifdef _DEBUG        
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")   //MAT processing
#pragma comment(lib, "opencv_objdetect249d.lib") //HOGDescriptor
#pragma comment(lib, "opencv_gpu249d.lib")
#pragma comment(lib, "opencv_features2d249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#else
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_objdetect249.lib")
#pragma comment(lib, "opencv_gpu249.lib")
#pragma comment(lib, "opencv_features2d249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#endif 

using namespace std;
using namespace cv;

#define WIDTH_DENSE (80)
#define HEIGHT_DENSE (60)

#define DENSE_DRAW 0 //dense optical flow arrow drawing or not
#define GLOBAL_MOTION_TH1 1
#define GLOBAL_MOTION_TH2 70


float drawOptFlowMap_gpu (const Mat& flow_x, const Mat& flow_y, Mat& cflowmap, int step, float scaleX, float scaleY, int drawOnOff);


int main()
{
 //stream /////////////////////////////////////////////////
 VideoCapture stream1("C:\\videoSample\\medical\\HUV-03-14.wmv"); 

 //variables /////////////////////////////////////////////
 Mat O_Img; //Mat
 gpu::GpuMat O_Img_gpu; //GPU
 gpu::GpuMat R_Img_gpu_dense; //gpu dense resize
 gpu::GpuMat R_Img_gpu_dense_gray_pre; //gpu dense resize gray
 gpu::GpuMat R_Img_gpu_dense_gray; //gpu dense resize gray
 gpu::GpuMat flow_x_gpu, flow_y_gpu;
 Mat flow_x, flow_y;

 //algorithm *************************************
 //dense optical flow
 gpu::FarnebackOpticalFlow fbOF;
 

 //running once //////////////////////////////////////////
 if(!(stream1.read(O_Img))) //get one frame form video
 {
  printf("Open Fail !!\n");
  return 0; 
 }

  //for rate calucation
 float scaleX, scaleY;
 scaleX = O_Img.cols/WIDTH_DENSE;
 scaleY = O_Img.rows/HEIGHT_DENSE;

 O_Img_gpu.upload(O_Img); 
 gpu::resize(O_Img_gpu, R_Img_gpu_dense, Size(WIDTH_DENSE, HEIGHT_DENSE));
 gpu::cvtColor(R_Img_gpu_dense, R_Img_gpu_dense_gray_pre, CV_BGR2GRAY);


 //unconditional loop   ///////////////////////////////////
 while (true) {
  //reading
  if( stream1.read(O_Img) == 0) //get one frame form video   
   break;

  // ---------------------------------------------------
  //upload cou mat to gpu mat
  O_Img_gpu.upload(O_Img); 
  //resize
  gpu::resize(O_Img_gpu, R_Img_gpu_dense, Size(WIDTH_DENSE, HEIGHT_DENSE));
  //color to gray
  gpu::cvtColor(R_Img_gpu_dense, R_Img_gpu_dense_gray, CV_BGR2GRAY);
  
  //calculate dense optical flow using GPU version
  fbOF.operator()(R_Img_gpu_dense_gray_pre, R_Img_gpu_dense_gray, flow_x_gpu, flow_y_gpu);
  flow_x_gpu.download( flow_x );
  flow_y_gpu.download( flow_y );


  //calculate motion rate in whole image
  float motionRate = drawOptFlowMap_gpu(flow_x, flow_y, O_Img, 1, scaleX, scaleY, DENSE_DRAW);
  //update pre image
  R_Img_gpu_dense_gray_pre = R_Img_gpu_dense_gray.clone();



  //display "moving" or "stop" message on the image.
  if(motionRate > GLOBAL_MOTION_TH2 ) //if motion generate over than 70%, this algorithm consider that video is moving.
  {
   char TestStr[100] = "Moving!!";
   putText(O_Img, TestStr, Point(30,60), CV_FONT_NORMAL, 2, Scalar(0,0,255),3,2); //OutImg is Mat class;   
  }else{
   char TestStr[100] = "Stop!!";
   putText(O_Img, TestStr, Point(30,60), CV_FONT_NORMAL, 2, Scalar(255,0,0),3,2); //OutImg is Mat class; 
  }


  // show image ----------------------------------------
  imshow("Origin", O_Img);   

  // wait key
  if( cv::waitKey(100) > 30)
   break;
 }
}



float drawOptFlowMap_gpu (const Mat& flow_x, const Mat& flow_y, Mat& cflowmap, int step, float scaleX, float scaleY, int drawOnOff)
{
 double count=0;

 float countOverTh1 = 0;
 int sx,sy;
 for(int y = 0; y < HEIGHT_DENSE; y += step)
 {
  for(int x = 0; x < WIDTH_DENSE; x += step)
  {
   
   if(drawOnOff)
   {
    Point2f fxy;    
    fxy.x = cvRound( flow_x.at< float >(y, x)*scaleX + x*scaleX );   
    fxy.y = cvRound( flow_y.at< float >(y, x)*scaleY + y*scaleY );   
    line(cflowmap, Point(x*scaleX,y*scaleY), Point(fxy.x, fxy.y), CV_RGB(0, 255, 0));   
    circle(cflowmap, Point(fxy.x, fxy.y), 1, CV_RGB(0, 255, 0), -1);   
   }

   float xx = fabs(flow_x.at< float >(y, x) );
   float yy = fabs(flow_y.at< float >(y, x) );

   float xxyy = sqrt(xx*xx + yy*yy);
   if( xxyy > GLOBAL_MOTION_TH1 )
    countOverTh1 = countOverTh1 +1;
   
   count=count+1;
  }
 }
 return (countOverTh1 / count) * 100;

}

