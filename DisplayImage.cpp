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
 * https://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
 * https://stackoverflow.com/questions/39013970/opencv-function-to-merge-mats-except-for-black-transparent-pixels
 * 
 */


// Function (forward) declaring
void adjustContrast(double alpha, cv::Mat &image);
void findFace(cv::Mat &image);
void blurEdge(cv::Mat &image);
int main(int argc, char** argv )
{
    //Original image
    cv::Mat image;
    
    //Error checking for image path
    if (argc != 2){
        printf("Image input not detected, using default.\n");
        image = cv::imread("viktor.jpg");
    }else{
        image = cv::imread( argv[1], 1 );
    }

    
    //Error checking for image
    if (!image.data){
        printf("No image data!\n");
        return -1;
    }
    
    //First copying over the original image into new_image
    cv::Mat new_image = image.clone();
    
    //Detecting face and drawing circle around it
    findFace(new_image);
    
    //Blur image
    blurEdge(new_image);
    
    //Boosting the contrast of the image
    adjustContrast(1.75,new_image);
    
    //Concat the two images to output
    cv::Mat composite_image;
    cv::hconcat(image,new_image,composite_image);
    
    //Print to screen
    cv::imshow("OpenCVTest", composite_image);

    cv::waitKey();
    return 0;
}

void adjustContrast(double alpha /* constrast adjustment factor*/, cv::Mat &image){
    image.convertTo(image,-1,alpha,0);
}

void findFace(cv::Mat &image){
    std::vector<cv::Rect> faces;//vector for storing region of interest (ROI) of faces
    
    //Convert to grayscale & equalize
    cv::Mat image_grey;
    cv::cvtColor(image,image_grey,cv::COLOR_BGR2GRAY);
    cv::equalizeHist(image_grey,image_grey);
    
    //Declaring and instantiating CascadeClassifier for facial detection
    cv::CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_alt.xml");

    //Detect the faces
    cascade.detectMultiScale(image_grey,faces,1.1,2,0|cv::CASCADE_SCALE_IMAGE,cv::Size(30,30)); 
    
    //Draw circles around faces
    for( size_t i = 0; i < faces.size(); i++ ){
        cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        cv::ellipse(image, center, 
                    cv::Size( faces[i].width*0.5, faces[i].height*0.5),
                    0, 0, 360,
                    cv::Scalar( 255, 0, 255 ),
                    4, 8, 0 );
    }
    
    std::cout << "Deteceted " << faces.size() << " faces!" << std::endl;
}

void blurEdge(cv::Mat &image){ 
    // A circle for mask, centered
    cv::Vec3f circ(image.cols/2,image.rows/2,image.rows/3);

    // Draw the mask: white circle on black background
    cv::Mat1b mask(image.size(), uchar(0));
    cv::circle(mask, cv::Point(circ[0], circ[1]), circ[2], cv::Scalar(255), cv::FILLED);

    // Create a black image for background
    cv::Mat3b res(image.size(),  cv::Vec3b(0,0,0));

    // Copy only the image under the white circle to black image
    image.copyTo(res, mask);
   
    //Blurring original image
    cv::blur(image,image,cv::Size(10,10));
    
    //Copying over the cropped picture, giving a Bokeh effect
    //Background is all blurred, foreground the the circle that we croppe with the mask
    //we are removing the black background, pasting the cropped picture 
    cv::Mat foreground;
    cv::inRange(res, cv::Scalar(0,0,0), cv::Scalar(0,0,0), foreground);
    res.copyTo(image, 255-foreground);
}


