#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

/*
 *Source code developed with reference to:
 * https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html
 * https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
 * 
 */

int main(int argc, char** argv )
{
    //Error checking for image path
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    //Original image
    cv::Mat image;
    image = cv::imread( argv[1], 1 );
    

    
    //Error checking for image
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    //First copying over the original image into new_image
    cv::Mat new_image = cv::Mat::zeros(image.size(),image.type());
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                new_image.at<cv::Vec3b>(y,x)[c] =image.at<cv::Vec3b>(y,x)[c];
            }
        }
    }
    
    //Boosting the contrast of the image
    double alpha = 1.5;//simple contrast variable
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                new_image.at<cv::Vec3b>(y,x)[c] =
                  cv::saturate_cast<uchar>( alpha*image.at<cv::Vec3b>(y,x)[c]);
            }
        }
    }
    
    //Concat the two images
    cv::Mat composite_image;
    cv::hconcat(image,new_image,composite_image);
    
    //cv::namedWindow("ModiFace-OpenCVTest", cv::WINDOW_AUTOSIZE );
    cv::imshow("OpenCVTest", composite_image);

    cv::waitKey(0);

    return 0;
}
