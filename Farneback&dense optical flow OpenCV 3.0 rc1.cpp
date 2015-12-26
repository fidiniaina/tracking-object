#include < iostream>  
#include < opencv2\opencv.hpp>  
#include < opencv2\highgui.hpp>  
//#include < opencv2\imgcodecs.hpp>  
#include < opencv2\videoio.hpp> 
#include < opencv2\core\cuda.hpp>
#include < opencv2\imgproc.hpp>
#include < opencv2\cudawarping.hpp>
#include < opencv2\cudaimgproc.hpp>
//#include < opencv2\cudaarithm.hpp>
#include < opencv2\cudaoptflow.hpp>


#ifdef _DEBUG             
#pragma comment(lib, "opencv_core300d.lib")     
#pragma comment(lib, "opencv_highgui300d.lib")  
//#pragma comment(lib, "opencv_imgcodecs300d.lib")  //imread
#pragma comment(lib, "opencv_videoio300d.lib") //video capture
#pragma comment(lib, "opencv_imgproc300d.lib") //line, circle
#pragma comment(lib, "opencv_cudawarping300d.lib") //cuda::resize
#pragma comment(lib, "opencv_cudaimgproc300.lib") //cuda::cvtcolor
#pragma comment(lib, "opencv_cudaarithm300d.lib") //cuda::farnebackOpticalFlow
#pragma comment(lib, "opencv_cudaoptflow300d.lib") 
#else     
#pragma comment(lib, "opencv_core300.lib")     
#pragma comment(lib, "opencv_highgui300.lib")  
//#pragma comment(lib, "opencv_imgcodecs300.lib")  //imread
#pragma comment(lib, "opencv_videoio300.lib") //video capture
#pragma comment(lib, "opencv_imgproc300.lib") // //line, circle
#pragma comment(lib, "opencv_cudawarping300.lib") //cuda::resize
#pragma comment(lib, "opencv_cudaimgproc300.lib") //cuda::cvtcolor
#pragma comment(lib, "opencv_cudaarithm300.lib") //cuda::farnebackOpticalFlow
#pragma comment(lib, "opencv_cudaoptflow300.lib") 

#endif      

using namespace std;
using namespace cv;

void drawOptFlowMap_gpu(const Mat& flow_xy, Mat& cflowmap, int step, const Scalar& color);

int main()
{

 int s = 1;

 unsigned long AAtime = 0, BBtime = 0;

 //variables  
 Mat GetImg, flow_x, flow_y, next, prvs, flow_xy;

 //gpu variable  
 cuda::GpuMat prvs_gpu, next_gpu, flow_x_gpu, flow_y_gpu, flow_xy_gpu;
 cuda::GpuMat prvs_gpu_o, next_gpu_o;
 cuda::GpuMat prvs_gpu_c, next_gpu_c;

 //file name  
 char fileName[100] = "M:\\____videoSample____\\Rendering\\Wildlife.avi";
 //video file open  
 VideoCapture stream1(fileName);   //0 is the id of video device.0 if you have only one camera
 if (!(stream1.read(GetImg))) //get one frame form video
  return 0;

 //gpu upload, resize, color convert
 prvs_gpu_o.upload(GetImg);

 
 cuda::resize(prvs_gpu_o, prvs_gpu_c, Size(GetImg.size().width / s, GetImg.size().height / s));
 cuda::cvtColor(prvs_gpu_c, prvs_gpu, CV_BGR2GRAY);

 //dense optical flow
 Ptr< cuda::FarnebackOpticalFlow > fbOF = cuda::FarnebackOpticalFlow::create();

 

 //unconditional loop
 while (true) {

  if (!(stream1.read(GetImg))) //get one frame form video     
   break;

   ///////////////////////////////////////////////////////////////////  
  //gpu upload, resize, color convert  
  next_gpu_o.upload(GetImg);
  cuda::resize(next_gpu_o, next_gpu_c, Size(GetImg.size().width / s, GetImg.size().height / s));
  cuda::cvtColor(next_gpu_c, next_gpu, CV_BGR2GRAY);
  ///////////////////////////////////////////////////////////////////  

  AAtime = getTickCount();
  //dense optical flow  
  fbOF->calc(prvs_gpu, next_gpu, flow_xy_gpu);

  BBtime = getTickCount();
  float pt = (BBtime - AAtime) / getTickFrequency();
  float fpt = 1 / pt;
  printf("%.2lf / %.2lf \n", pt, fpt);

  //copy for vector flow drawing  
  Mat cflow;
  resize(GetImg, cflow, Size(GetImg.size().width / s, GetImg.size().height / s));  
  flow_xy_gpu.download(flow_xy);
  drawOptFlowMap_gpu(flow_xy, cflow, 10, CV_RGB(0, 255, 0));
  imshow("OpticalFlowFarneback", cflow);

  ///////////////////////////////////////////////////////////////////  
  //Display gpumat  
  next_gpu.download(next);
  prvs_gpu.download(prvs);
  imshow("next", next);
  imshow("prvs", prvs);

  //prvs mat update  
  prvs_gpu = next_gpu.clone();

  if (waitKey(5) >= 0)
   break;
 }


 return 0;
}

void drawOptFlowMap_gpu(const Mat& flow_xy, Mat& cflowmap, int step, const Scalar& color)
{

 for (int y = 0; y < cflowmap.rows; y += step)
  for (int x = 0; x < cflowmap.cols; x += step)
  {
   Point2f fxy;
   fxy.x = cvRound(flow_xy.at< Vec2f >(y, x)[0] + x);
   fxy.y = cvRound(flow_xy.at< Vec2f >(y, x)[1] + y);


   cv::line(cflowmap, Point(x, y), Point(fxy.x, fxy.y), color);
   cv::circle(cflowmap, Point(fxy.x, fxy.y), 1, color, -1);

  }

}




