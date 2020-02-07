#include <stdio.h>


#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
/*
 *Source code developed with reference to:
 * https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html
 * https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
 * https://www.codementor.io/@shashwatjain661/how-detect-faces-using-opencv-and-python-c-nwyssng68
 */


// Function (forward) declaring
void boostContrast(double alpha, cv::Mat &image);
void findFace(cv::Mat &image);
void blurEdge(cv::Mat &image);

int main(int argc, char** argv )
{
    //Original image
    cv::Mat image;
    
    
    //Error checking for image path
    if ( argc != 2 )
    {
        printf("Image input not detected, using default.\n");
        image = cv::imread("viktor.jpg");
    }else{
        image = cv::imread( argv[1], 1 );
    }

    
    //Error checking for image
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    //First copying over the original image into new_image
    cv::Mat new_image = image.clone();
    
    //Boosting the contrast of the image
    boostContrast(1.5,new_image);
    
    //Detecting face and blurring
    findFace(new_image);
    
    //Blur image
    blurEdge(new_image);
    
    //Concat the two images to output
    cv::Mat composite_image;
    cv::hconcat(image,new_image,composite_image);
    
    //Print to screen
    cv::imshow("OpenCVTest", composite_image);

    cv::waitKey();
    return 0;
}

void findFace(cv::Mat &image){
    //convert to grayscale for ease of detection
    std::vector<cv::Rect> faces;
    cv::Mat image_grey;
    cv::cvtColor(image,image_grey,cv::COLOR_BGR2GRAY);
    
    
    //Declaring and instantiating CascadeClassifier for facial detection
    cv::CascadeClassifier cascade;
    cascade.load("haarcascade_frontalcatface_extended.xml");
    cascade.detectMultiScale(image_grey,faces);
    
    //read for drawing
    cv::Scalar color = cv::Scalar(0, 0, 255);
    
    //Draw circle around the face(s)
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        cv::Rect r = faces[i];
        cv::Point center; 
        int radius;
        
        center.x = cvRound((r.x + r.width*0.5)); 
        center.y = cvRound((r.y + r.height*0.5)); 
        radius = cvRound((r.width + r.height)); 
        circle( image, center, radius, color, 3, 8, 0 ); 
    }

    
    std::cout << "deteceted " << faces.size() << "faces" << std::endl;
    
}

void boostContrast(double alpha, cv::Mat &image){
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                //multiplying the original image by the constrast alpha factor
                image.at<cv::Vec3b>(y,x)[c] =
                  cv::saturate_cast<uchar>( alpha*image.at<cv::Vec3b>(y,x)[c]);
            }
        }
    }
}

void blurEdge(cv::Mat &image){
    //Blur background
    //cv::Mat background = image.clone();
    //cv::blur(background,background,cv::Size(10,10));
    
    cv::Rect region(0, 0, 50, image.rows);
    cv::GaussianBlur(image(region), image(region), cv::Size(0, 0), 4);
    //Re apply the image ontop
    //cv::GaussianBlur(image, background, cv::Size(0, 0), 3);
    //cv::addWeighted(image, 1.5, background, -0.5, 0, background);
    
    /*
    //First convert to greyscale
    cv::Mat image_grey;
    cv::cvtColor(image,image_grey,cv::COLOR_BGR2GRAY);
    //Prepare mask
    cv::Mat mask;
    image_grey.convertTo(mask,CV_32FC1);
    cv::threshold(1.0-mask,mask,0.9,1.0,cv::THRESH_BINARY_INV);
    
    //Apply blur on the mask
    cv::GaussianBlur(mask,mask,cv::Size(21,21),11.0);
    
    //Get the image with mask
    cv::Mat bg= cv::Mat(image.size(),CV_32FC3);
    cv::Mat result;
    std::vector<cv::Mat> ch_img(3);
    std::vector<cv::Mat> ch_bg(3);
    cv::split(image,ch_img);
    cv::split(bg,ch_bg);
    
    ch_img[0]=ch_img[0].mul(mask)+ch_bg[0].mul(1.0-mask);
    ch_img[1]=ch_img[1].mul(mask)+ch_bg[1].mul(1.0-mask);
    ch_img[2]=ch_img[2].mul(mask)+ch_bg[2].mul(1.0-mask);
    
    cv::merge(ch_img,result);
    cv::merge(ch_bg,bg);
    imshow("result",result);
    */
}
