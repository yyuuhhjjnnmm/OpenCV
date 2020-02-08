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
 * 
 * 
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
    
    //Detecting face and draw circle
    findFace(new_image);
    
    //Boosting the contrast of the image
    //boostContrast(1.5,new_image);
    new_image.convertTo(new_image,-1,2,0);
    
    
    
    //Blur image
    //blurEdge(new_image);
    
    //Concat the two images to output
    cv::Mat composite_image;
    cv::hconcat(image,new_image,composite_image);
    
    //Print to screen
    cv::imshow("OpenCVTest", composite_image);

    cv::waitKey();
    return 0;
}

void findFace(cv::Mat &image){
    //convert to grayscale for ease of detection & equalize
    std::vector<cv::Rect> faces;
    cv::Mat image_grey;
    cv::cvtColor(image,image_grey,cv::COLOR_BGR2GRAY);
    cv::equalizeHist(image_grey,image_grey);
    
    //Declaring and instantiating CascadeClassifier for facial detection
    cv::CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_alt.xml");

    //Detect the faces
    cascade.detectMultiScale(image_grey,faces,1.1,2,0|cv::CASCADE_SCALE_IMAGE,cv::Size(30,30)); 
    
    //Draw circles around faces
    for( size_t i = 0; i < faces.size(); i++ )
    {
        cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        cv::ellipse( image, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );

        cv::Mat faceROI = image_grey( faces[i] );
        std::vector<cv::Rect> eyes;
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

void blurEdge(cv::Mat&image){
    cv::Mat background = image.clone();
    cv::blur(background,background,cv::Size(10,10));
    cv::Mat cropedImage = background(cv::Rect(50,50,200,220));
    cv::GaussianBlur(cropedImage, image, cv::Size(0, 0), 3);
    cv::addWeighted(cropedImage, -0.5, image, 1.5, 0, image);
    cv::imwrite("background.png", cropedImage);

}

// void blurEdge(cv::Mat&image){
//     //Draw a circle for mask
//     cv::Point center; 
//     center.x = image.cols/2;
//     center.y = image.rows/2;
//     cv::Mat mask = cv::Mat::zeros(image.rows,image.cols,CV_8UC1);
//     circle(mask,center,image.rows/3,cv::Scalar(255,255,255),-1,8,0);
//     
//     //Copy the image under mask
//     cv::Mat result(image.size(),CV_8UC1, cv::Scalar(0, 0, 0));
//     image.copyTo(result,mask);
//     
//     
// 
// //     cv::Mat temp,alpha;
// //     cv::Mat dest(image.rows,image.cols,CV_8UC4);
// //     cv::cvtColor(result,temp,cv::COLOR_BGR2GRAY);
// //     cv::threshold(temp,alpha,100,255,cv::THRESH_BINARY);
// //     cv::Mat rgb[3];
// //     cv::split(result,rgb);
// //     cv::Mat rgba[4]={rgb[0],rgb[1],rgb[2],alpha};
// //     cv::merge(rgba,4,dest);
//     
//     
//     cv::imwrite("result.png", dest);
//     
//     
//     //Blur background
//     cv::blur(image,image,cv::Size(10,10));
//     
//     //Reapply the cropped ROI
//     cv::addWeighted(image,0.5, result, .5, 0.0, image);
// }

/*
void blurEdge(cv::Mat &image){
    //Blur background
    cv::Mat background = image.clone();
    cv::blur(background,background,cv::Size(10,10));
    
    //cv::Rect region(0, 0, 50, image.rows);
    //cv::GaussianBlur(image(region), image(region), cv::Size(0, 0), 4);
    

    cv::Vec3f circ(background.cols/2,background.rows/2,background.cols/2);//Hough circle
    
    
    //Daw the mask
    //cv::Mat1b mask(background.size(),uchar(0));
    //circle(mask, cv::Point(circ[0],circ[1]),circ[2], cv::Scalar(255),cv::FILLED);
    
    //Bounding box
    cv::Rect bbox(circ[0] - circ[2], circ[1] - circ[2], 2 * circ[2], 2 * circ[2]);
    
    
    //Crop
    //background.copyTo(background,mask);
    //background = background(bbox);
    
    //Re apply the image ontop
    background = background(bbox);
    cv::imwrite("background.png", background);
    cv::addWeighted(image, 0, background, 1, 0.0, image);
    
    //cv::GaussianBlur(image, background, cv::Size(0, 0), 3);
    
    
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
    
}*/

