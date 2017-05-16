#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat img,mask,temp_frame; //test image and the mask for the grab cut segmentation
Mat segmented,result;
Mat fgdModel,bgdModel;
Mat result_gray;

bool lbutton_down=false,lbutton_up=false; //check the status of the mouse buttons
bool rectSet=false,labelSet=false;

Point corner1,corner2; //store the corners for the ROI
Rect ROI_rect;

int BGD_FLAG=CV_EVENT_FLAG_CTRLKEY; //flag for labeling BGD pixels
int FGD_FLAG=CV_EVENT_FLAG_SHIFTKEY;//flag for labeling FGD pixels
int BGD_COLOR=GC_BGD,FGD_COLOR=GC_FGD;
int connectedComponentIndex=99;

vector<int> component_length;
vector<vector<Point2f> > roadPoints;

string window_name="image";
string seg_window="segmented";

const Scalar BLUE_COLOR=Scalar(255,0,0);
const Scalar GREEN_COLOR=Scalar(0,255,0);
const int ROI_SIZE=60;

Mat nextIteration(Mat input,int mode){

    int num_iterations=1;
    grabCut(input,mask,ROI_rect,bgdModel,fgdModel,num_iterations,mode);
    //get the foreground pixels
    Mat fore_result;
    compare(mask,GC_PR_FGD,fore_result,CMP_EQ);
    Mat segmented(img.size(),CV_8UC3,Scalar(255,255,255));
    img.copyTo(segmented,fore_result);
    return segmented;
}
void segMouseCallback(int event,int x,int y,int flags,void*){
    if(event == EVENT_LBUTTONDOWN){
        lbutton_down=true;
    }
    if(event == EVENT_LBUTTONUP){
        lbutton_down=false;
    }
    if(lbutton_down){
        if(flags & BGD_FLAG){
            circle(temp_frame,Point(x,y),5.0,BLUE_COLOR,CV_FILLED);
            circle(mask,Point(x,y),5.0,BGD_COLOR,CV_FILLED);
        }
        if(flags & FGD_FLAG){
            circle(temp_frame,Point(x,y),5.0,GREEN_COLOR,CV_FILLED);
            circle(mask,Point(x,y),5.0,FGD_COLOR,CV_FILLED);
        }
        imshow(seg_window,temp_frame);
    }
}
void mousePointer(int event,int x,int y,int flags,void*){

    if(event == EVENT_LBUTTONDOWN){
        //record the corner 1 values
        corner1.x=x;
        corner1.y=y;
        cout << corner1 << endl;
        lbutton_down=true;
    }
    if(event == EVENT_LBUTTONUP){
        //record the second corner and get the ROI
        //record the corner 1 values
        corner2.x=x;
        corner2.y=y;
        cout << corner2 << endl;
        lbutton_up=true;
    }
    if(lbutton_down && !lbutton_up){
        //update the rectangle region
        Mat local;
        img.copyTo(local);
        Point temp;
        temp.x=x;
        temp.y=y;
        rectangle(local,corner1,temp,Scalar(0,0,255),2);
        imshow(window_name,local);
    }
    if(lbutton_down && lbutton_up){

        ROI_rect.width=abs(corner1.x-corner2.x);
        ROI_rect.height=abs(corner1.y-corner2.y);
        ROI_rect.x=min(corner1.x,corner1.x);
        ROI_rect.y=min(corner1.y,corner2.y);
        cout << ROI_rect <<endl;

        //set the status of the rectangle flag
        rectSet=true;
        segmented=nextIteration(img,GC_INIT_WITH_RECT);
        temp_frame=segmented.clone();
        namedWindow(seg_window);
        imshow(seg_window,segmented);
        setMouseCallback(seg_window,segMouseCallback);

        lbutton_down=false;
        lbutton_up=false;
    }
}
void FindBlobs(const Mat &binary, vector < vector<Point2f> > &blobs)
{
    blobs.clear();
    Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            Rect rect;
            floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);

            vector <Point2f> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(Point2i(j,i));
                }
            }
            blobs.push_back(blob);

            label_count++;
        }
    }
}
int findLargestComponentIndex(vector<vector<Point2f> > sample){
    int largestComponent=sample[0].size();
    int index=0;
    for(int i=1;i<sample.size();i++){
        if(largestComponent < sample[i].size()){
            largestComponent=sample[i].size();
            index=i;
        }
    }
    return index;
}
Point getSkeletonPoint(Mat frame){

    //get the skeleton of the frame
    /*Mat skel(frame.size(), CV_8UC1, Scalar(0));
    Mat temp;
    Mat eroded;

    Mat element = cv::getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat element2= cv::getStructuringElement(MORPH_RECT,Size(3,3));
    //Mat element3=cv::getStructuringElement(MORPH_RECT,Size(5,5));

    bool done;
    do
    {
      erode(frame, eroded, element);
      dilate(eroded, temp, element); // temp = open(img)
      subtract(frame, temp, temp);
      bitwise_or(skel, temp, skel);
      eroded.copyTo(frame);

      done = (countNonZero(frame) == 0);
    } while (!done);
    dilate(skel,skel,element2);
    //get the largest connected component
    Mat bin_skel;
    threshold(skel,bin_skel,0.0,1.0,CV_THRESH_BINARY);
    vector<vector<Point2f> > points;
    FindBlobs(bin_skel,points);
    //find the largest component
    int index=findLargestComponentIndex(points);
    vector<Point2f> largestBlob=points[index];
    for(int i=0;i<largestBlob.size();i++){
        Point2f temp_point=largestBlob[i];
        skel.at<unsigned char>(temp_point.y,temp_point.x)=connectedComponentIndex;
    }
    compare(skel,connectedComponentIndex,skel,CMP_EQ);
    //find the first white point of the skeleton
    Point position;
    int counter=0;
    for(int i=skel.rows-1;i>=0;i--){
        for(int j=skel.cols-1;j>=0;j--){
            int val=skel.at<unsigned char>(i,j);
            //cout << val << endl;
            //counter+=1;
            if(val == 255){
                position.x=j;
                position.y=i;
                break;
            }
        }
    }*/


    //cout << position << endl;
    //return position;
}
Mat preprocess_frame(){

    Mat binary;

    vector < vector<Point2f> > blobs,holes;

    cvtColor(result,result_gray,CV_BGR2GRAY);
    //apply the threshold
    threshold(result_gray,result_gray,0,255,CV_THRESH_OTSU);
    bitwise_not(result_gray,result_gray);
    //namedWindow("Thresh result");
    //imshow("Thresh result",result_gray);

    threshold(result_gray, binary, 0.0, 1.0, THRESH_BINARY);
    Mat output=Mat::zeros(result_gray.size(),CV_8UC3);

    FindBlobs(binary,blobs);
    // Randomly color the blobs
    for(size_t i=0; i < blobs.size(); i++) {
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            output.at<Vec3b>(y,x)[0] = b;
            output.at<Vec3b>(y,x)[1] = g;
            output.at<Vec3b>(y,x)[2] = r;
        }
    }
    namedWindow("Connected Components");
    imshow("Connected Components",output);
    //find the largest connected component
    int index=findLargestComponentIndex(blobs);
    vector<Point2f> largestBlob=blobs[index];

    for(int i=0;i<largestBlob.size();i++){
        Point2f temp_point=largestBlob[i];
        result_gray.at<unsigned char>(temp_point.y,temp_point.x)=connectedComponentIndex;
    }
    compare(result_gray,connectedComponentIndex,result_gray,CMP_EQ);

    //perform hole filling in the road component
    bitwise_not(result_gray,result_gray);
    Mat binary_img(result_gray.size(),CV_8UC1);
    threshold(result_gray,binary_img,0.0,1.0,THRESH_BINARY);
    FindBlobs(binary_img,holes);

    //find the largest connected component
    index=findLargestComponentIndex(holes);
    vector<Point2f> temp;
    for(int i=0;i<holes.size();i++){
        if(i!=index){
            temp=holes[i];
            for(int j=0;j<temp.size();j++)
                result_gray.at<unsigned char>(temp[j].y,temp[j].x)=0;
        }
    }
    namedWindow("Hole filled");
    imshow("Hole filled",result_gray);
    Mat output_result(img.size(),CV_8UC3);
    img.copyTo(output_result,result_gray);
    namedWindow("road_results");
    imshow("road_results",result_gray);
    bitwise_not(result_gray,result_gray);

    namedWindow("new result");
    imshow("new result",result_gray);

    threshold(result_gray,result_gray,0.0,1.0,THRESH_BINARY);
    FindBlobs(result_gray,roadPoints);

    return output_result;
}
vector<KeyPoint> computeFastFeatures(Mat);
int main(int argc, char *argv[])
{
    /* Initial road segmentation performed for system initialization */
    //load the main image
    img = imread("img_005.png", CV_LOAD_IMAGE_COLOR);
    namedWindow(window_name);
    imshow(window_name,img);
    setMouseCallback(window_name,mousePointer);

    int code;
    if((code=waitKey(0))=='n'){
        result=nextIteration(segmented,GC_INIT_WITH_MASK);
    }
    //namedWindow("result");
    //imshow("result",result);
    Mat road;
    road=preprocess_frame();
    namedWindow("final_result");
    imshow("final_result",road);

    while(waitKey(0)!='q'){}

    destroyAllWindows();
    cout << "Road tracking will now begin...." <<endl;
    Mat MatRoadPoints=Mat(roadPoints[0]);
    /*Tracking of road performed to detect the road in successive images */
    VideoCapture cap("a.avi");
    if(!cap.isOpened())
        cout << "The video file could not be opened" <<endl;

    Mat prev_frame=img.clone();
    Mat next_frame;

    vector<Point2f> prevPointFeatures,nextPointFeatures;
    vector<Point2f> projectedPoints;
    vector<Point2f> currentPoints=roadPoints[0];
    vector<KeyPoint> prevFeatures,nextFeatures;
    Mat nextFrameKeypoints;

    prevFeatures=computeFastFeatures(prev_frame);
    Mat prev_temp_frame,next_temp_frame;
    Mat updated_frame;
    Mat homography_matrix;
    int counter=0;

    while(waitKey(1)!='q' && cap.isOpened()){

        /* Project the common road part from the road initialization onto successive frames*/

        //draw the previous frames on the new frames
        drawKeypoints(prev_frame,prevFeatures,prev_temp_frame,BLUE_COLOR);
        namedWindow("feature window");
        imshow("feature window",prev_temp_frame);

        //get the next frame for the optical flow calculation
        cap >> next_frame;
        if(next_frame.empty())
            break;

        KeyPoint::convert(prevFeatures,prevPointFeatures);
        Mat status;
        Mat err;
        //get the optical flow for the image features
        calcOpticalFlowPyrLK(prev_frame,next_frame,Mat(prevPointFeatures),nextFrameKeypoints,status,err);

        for(int i=0;i<nextFrameKeypoints.rows;i++)
            for(int j=0;j<nextFrameKeypoints.cols;j++)
                nextPointFeatures.push_back(nextFrameKeypoints.at<Point2f>(i,j));

        KeyPoint::convert(nextPointFeatures,nextFeatures);
        //draw the features on the next frame
        drawKeypoints(next_frame,nextFeatures,next_temp_frame,GREEN_COLOR);
        namedWindow("feature_image2");
        imshow("feature_image2",next_temp_frame);

        //compute the Homography matrix for the two frames
        homography_matrix=findHomography(Mat(prevPointFeatures),Mat(nextPointFeatures),CV_RANSAC);
        //project the points
        perspectiveTransform(currentPoints,projectedPoints,homography_matrix);
        currentPoints=projectedPoints;

        //update the road in the next frame
        updated_frame = next_frame.clone();
        Mat updated_frame_bin(updated_frame.size(),CV_8UC1,Scalar(0,0,0));
        for(int i=0;i<projectedPoints.size();i++){
            Point2f tmp_point=projectedPoints[i];
            if(tmp_point.y > updated_frame.rows)
                break;
            updated_frame.at<Vec3b>(floor(tmp_point.y),floor(tmp_point.x))={0,0,0};
            updated_frame_bin.at<unsigned char>(floor(tmp_point.y),floor(tmp_point.x))=255;
        }
        namedWindow("updated_frame_bin");
        imshow("updated_frame_bin",updated_frame_bin);
        /* Detect the new road part in the successive frames */
        //do this every 5th frame
        if(counter==5){

            Point skel_center=getSkeletonPoint(updated_frame_bin);
            Rect ROI_new;
            ROI_new.height=ROI_SIZE;
            ROI_new.width=ROI_SIZE;
            ROI_new.x=skel_center.x-ROI_SIZE/2;
            ROI_new.y=skel_center.y;

            //perform GrabCut on the ROI
            grabCut(prev_frame,mask,ROI_new,bgdModel,fgdModel,1,GC_INIT_WITH_RECT);
            Mat fore_result2;
            compare(mask,GC_PR_FGD,fore_result2,CMP_EQ);
            //update the overall frame(new road and the old road parts)
            Mat new_road_bin(prev_frame.size(),CV_8UC1);
            bitwise_or(fore_result2,updated_frame_bin,new_road_bin);
            Mat new_road(prev_frame.size(),CV_8UC3,Scalar(0,0,0));
            prev_frame.copyTo(new_road,new_road_bin);
            //cout << ROI_new << endl;

            //display the ROI on the original image
            //rectangle(road,ROI_new,Scalar(0,0,255),1);
            //namedWindow("ROIresult");
            //imshow("ROIresult",road);
            updated_frame=new_road.clone();
            counter=0;
        }
        namedWindow("updated road");
        imshow("updated road",updated_frame);
        prev_frame=next_frame.clone();
        prevFeatures=computeFastFeatures(prev_frame);
        nextPointFeatures.clear();
        //counter+=1;

    }
    waitKey(0);
    return 0;
}
vector<KeyPoint> computeFastFeatures(Mat frame){
    Mat keypoint_image;
    vector<KeyPoint> keypoint_descriptors;
    int threshold=50;
    FAST(frame,keypoint_descriptors,threshold,true);
    return keypoint_descriptors;
}
