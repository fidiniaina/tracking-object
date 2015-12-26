#include < stdio.h>
#include < iostream>

#include < opencv2\opencv.hpp>
#include < opencv2/core/core.hpp>
#include < opencv2/highgui/highgui.hpp>
//#include < opencv2/video/background_segm.hpp>
#include < opencv2\gpu\gpu.hpp>
#include < opencv2\stitching\detail\matchers.hpp >  
//#include < opencv2\nonfree\features2d.hpp >    


#ifdef _DEBUG        
#pragma comment(lib, "opencv_core247d.lib")
#pragma comment(lib, "opencv_imgproc247d.lib")   //MAT processing
//#pragma comment(lib, "opencv_objdetect247d.lib") //HOGDescriptor
#pragma comment(lib, "opencv_gpu247d.lib")
#pragma comment(lib, "opencv_features2d247d.lib")
#pragma comment(lib, "opencv_highgui247d.lib")
//#pragma comment(lib, "opencv_ml247d.lib")
//#pragma comment(lib, "opencv_stitching247d.lib");
#pragma comment(lib, "opencv_nonfree247d.lib")
//#pragma comment(lib, "opencv_video247d.lib")
#else
#pragma comment(lib, "opencv_core247.lib")
#pragma comment(lib, "opencv_imgproc247.lib")
//#pragma comment(lib, "opencv_objdetect247.lib")
#pragma comment(lib, "opencv_gpu247.lib")
#pragma comment(lib, "opencv_features2d247.lib")
#pragma comment(lib, "opencv_highgui247.lib")
//#pragma comment(lib, "opencv_ml247.lib")
//#pragma comment(lib, "opencv_stitching247.lib");
#pragma comment(lib, "opencv_nonfree247.lib");
//#pragma comment(lib, "opencv_video247d.lib")
#endif 



using namespace cv;
using namespace std;


static void download(const gpu::GpuMat& d_mat, vector< Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void main()
{

 

 gpu::GpuMat img1(imread("C:\\videoSample\\Image\\Picture6.jpg", CV_LOAD_IMAGE_GRAYSCALE)); 
    gpu::GpuMat img2(imread("C:\\videoSample\\Image\\Picture7.jpg", CV_LOAD_IMAGE_GRAYSCALE)); 


 unsigned long t_AAtime=0, t_BBtime=0;
 float t_pt;
 float t_fpt;
 t_AAtime = getTickCount(); 

 gpu::GoodFeaturesToTrackDetector_GPU GFTTDetector_gpu(200); //(8000,0.01, 0.0)
 gpu::PyrLKOpticalFlow OpticalPYLK_gpu;
 gpu::GpuMat leftCorners;
 gpu::GpuMat rightCorners;

 //Feature extraction
 GFTTDetector_gpu(img1, leftCorners);
 gpu::GpuMat status_gpu;
 gpu::GpuMat err_gpu; //right feature extraction and matching
 OpticalPYLK_gpu.sparse(img1, img2, leftCorners, rightCorners, status_gpu, &err_gpu);

 vector< Point2f> leftPts(leftCorners.cols);
 vector< Point2f> rightPts(rightCorners.cols);
 download(leftCorners, leftPts);
 download(rightCorners, rightPts);

 
 Mat status;
 status_gpu.download(status);
 Mat error;
 err_gpu.download(error);

 //cout << status.size() << endl;
 //1st matching filter
 vector< Point2f> right_to_find;
 vector< int> right_to_find_back_idx;
 //select good matching point from status.
 for(size_t m=0; m< status.cols; ++m)
 {
  int stat = status.at< unsigned char >(0,m);
  float err = error.at< float >(0,m);
  
  //&& err < 12
  if( stat ){
   right_to_find_back_idx.push_back(m);
   right_to_find.push_back( rightPts[m] );
  }
  status.at< unsigned char >(0,m) = 0;
 }

 

 //2nd matching filter
 std::set< int> found_in_right_points;
 vector< DMatch> good_matches;
 Mat right_point_to_find_flat = Mat(right_to_find).reshape(1,right_to_find.size()); 
 Mat right_features_flat = Mat(rightPts).reshape(1,rightPts.size()); 

 //cout << right_point_to_find_flat.size() << right_to_find.size() << endl;

 gpu::GpuMat Step2_Rpt(right_point_to_find_flat); //
 gpu::GpuMat Step1_Rpt(right_features_flat);
 vector< vector< DMatch> > knn_matches;
 gpu::BruteForceMatcher_GPU< L2< float> > matcher; 
 matcher.radiusMatch(Step2_Rpt, Step1_Rpt, knn_matches, 2.0f);

 for(int i=0;i< knn_matches.size();i++) {
  DMatch _m;
  if(knn_matches[i].size()==1) {
   _m = knn_matches[i][0];
  } else if(knn_matches[i].size()>1) {
   if(knn_matches[i][0].distance / knn_matches[i][1].distance < 0.7) {
    _m = knn_matches[i][0];
   } else {
    continue; // did not pass ratio test
   }
  } else {
   continue; // no match
  }

  // prevent duplicates
  if (found_in_right_points.find(_m.trainIdx) == found_in_right_points.end()) { 
   _m.queryIdx = right_to_find_back_idx[_m.queryIdx]; //back to original indexing of points for < i_idx >
   good_matches.push_back(_m);
   found_in_right_points.insert(_m.trainIdx);
  }
 }


 t_BBtime = getTickCount();
 t_pt = (t_BBtime - t_AAtime)/getTickFrequency();
 t_fpt = 1/t_pt;
 printf("%.4lf sec/ %.4lf fps\n",  t_pt, t_fpt );
 

 //draw feature and matching
 Mat img11, img22;
 img1.download(img11);
 img2.download(img22);
 
 Mat canvas = Mat::zeros(img11.rows, img11.cols*2, img11.type() );
 img11.copyTo(canvas(Range::all(), Range(0, img11.cols) ) );
 img22.copyTo(canvas(Range::all(), Range(img11.cols, img11.cols*2) ));
 cvtColor(canvas, canvas, CV_GRAY2BGR);


 vector< Point2f > baseMatch, targetMatch;
 for(int i=0;i< good_matches.size();i++) {
  baseMatch.push_back( leftPts[good_matches[i].queryIdx]  );
  targetMatch.push_back( rightPts[good_matches[i].trainIdx] );
 }
 for(int d=0; d< baseMatch.size(); ++d)
  {
   int x1 = leftPts[d].x;
   int y1 = leftPts[d].y;

   int x2 = img11.cols +rightPts[d].x;
   int y2 = rightPts[d].y;

   cv::circle(canvas, Point( x1, y1 ), 3, CV_RGB(255,0,0),2);
   cv::circle(canvas, Point( x2, y2 ), 3, CV_RGB(255,0,0),2);
   cv::line( canvas, Point(x1, y1), Point(x2, y2), CV_RGB(0,0,255) );
  }



 namedWindow("matches", 0);
    imshow("matches", canvas);

 waitKey(0);

}
