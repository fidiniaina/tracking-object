#include < stdio.h>    
  
#include < opencv2\opencv.hpp>    
#include < opencv2/core/core.hpp>    
#include < opencv2/highgui/highgui.hpp>    
#include < opencv2\nonfree\features2d.hpp >        
  
  
  
#ifdef _DEBUG            
#pragma comment(lib, "opencv_core249d.lib")    
#pragma comment(lib, "opencv_imgproc249d.lib")   //MAT processing    
#pragma comment(lib, "opencv_objdetect249d.lib") //HOGDescriptor    
#pragma comment(lib, "opencv_features2d249d.lib")    
#pragma comment(lib, "opencv_highgui249d.lib")    
#pragma comment(lib, "opencv_video249d.lib")    
#else    
#pragma comment(lib, "opencv_core249.lib")    
#pragma comment(lib, "opencv_imgproc249.lib")    
#pragma comment(lib, "opencv_objdetect249.lib")    
#pragma comment(lib, "opencv_features2d249.lib")    
#pragma comment(lib, "opencv_highgui249.lib")   
#pragma comment(lib, "opencv_video249.lib")    
#endif     
  
using namespace std;    
using namespace cv; 

void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color);

void main()
{
 Mat prev_im,nex_im,flow;

 prev_im=imread("1.jpg",0);
 imshow("previm",prev_im);


 nex_im=imread("2.jpg",0);
 imshow("nextim",nex_im);


 calcOpticalFlowFarneback(prev_im,nex_im,flow,0.5, 1, 12, 2, 7, 1.5, 0); 

 Mat cflow;
 cvtColor(prev_im, cflow, CV_GRAY2BGR);
 drawOptFlowMap(flow, cflow, 10, CV_RGB(0, 255, 0));  


 imshow("OpticalFlowFarneback", cflow);
 waitKey(0);

}


void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {  
 for(int y = 0; y < cflowmap.rows; y += step)  
  for(int x = 0; x < cflowmap.cols; x += step)  
  {  
   const Point2f& fxy = flow.at< Point2f>(y, x);  
   line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),  
    color);  
   circle(cflowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);  
  }  
}  

