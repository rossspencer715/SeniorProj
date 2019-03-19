//
//  OpenCVWrapper.mm
//  Senior Project
//
//  Created by Ross Spencer.
//  Copyright © 2019 Ross Spencer. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import "OpenCVWrapper.h"
#import <opencv2/imgcodecs/ios.h>
#import "cpp/native-lib.cpp"

//C++ implementations of header methods
@implementation OpenCVWrapper

+ (NSString *)openCVVersionString {
    return [NSString stringWithFormat:@"OpenCV Version %s",  CV_VERSION];
}

+ (UIImage *)makeGray:(UIImage *)image {
    //transform UIImage to cv::mat
    cv::Mat imageMat;
    UIImageToMat(image, imageMat);
    
    //if image already greyscale, return it
    if(imageMat.channels() == 1){
        return image;
    }
    
    //transform cvMat file to greyscale
    cv::Mat greyMat;
    cv::cvtColor(imageMat, greyMat, cv::COLOR_BGR2GRAY);
    
    //transform greyMat to UIImage
    return MatToUIImage(greyMat);
    
}

+ (NSArray *)preprocessWrapper:(UIImage *)image {
    //want to make a mat
    cv::Mat imageMat;
    UIImageToMat(image, imageMat);
    
    //call the preprocess function??
    std::vector<double> ans = preprocess(&imageMat);
    NSArray * idk;
    return idk;
}

@end
