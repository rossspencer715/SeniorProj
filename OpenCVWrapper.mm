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
//#import "cpp/native-lib.cpp"

//#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <algorithm>
#include "dirent.h"
#include <fstream>
#include <vector>
#include <iterator>
#include <random>
#include <list>
#include "math.h"
#include "utils.h"
#include "nanoflann.hpp"

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




//Template average RGB values
const double TEMPLATE_R = 21;
const double TEMPLATE_G = 74;
const double TEMPLATE_B = 177;

const int NUM_ITERATIONS = 7;
const int NUM_SUPERPIXELS = 700;
const int NUM_LEVELS = 2;
const int SUPERPIXEL_PRIOR = 3;

const double GAMMA_DECODE = 2.2;
const double GAMMA_ENCODE = 0.5;
const double COLOR_MAX = 255;




//preprocesses data
std::vector<double> preprocess(cv::Mat* inputAddr){
    
    using namespace std;
    //using namespace cv;
    using namespace cv::ximgproc;
    
    cv::Mat img, cvImg, result, mask, ycbcr_im, superpixel_labels, red_im, green_im, blue_im, labels, thresh_img, saddle;
    int width, height, superpixel_num, img_total;
    cv::Ptr<SuperpixelSEEDS> seeds;
    
    int num_iterations = 7;
    int num_superpixels = 700;
    int num_levels = 2;
    
    cv::Mat* pInputImage = (cv::Mat*)inputAddr;
    img = *pInputImage;
    img.convertTo(cvImg, CV_8U);
    width = cvImg.size().width;
    height = cvImg.size().height;
    
    //Initialize super pixels
    seeds = createSuperpixelSEEDS(width, height, cvImg.channels(), num_superpixels, num_levels, 3);
    result = cvImg;
    saddle = cvImg;
    seeds->iterate(result, num_iterations);
    seeds->getLabelContourMask(mask, false);
    
    //Create average super pixel color image
    superpixel_num = seeds->getNumberOfSuperpixels(); //Number to iterate over
    seeds->getLabels(superpixel_labels);
    seeds->getLabels(labels);
    
    vector<cv::Mat> channels;
    split(img, channels);
    red_im = channels[2];
    green_im = channels[1];
    blue_im = channels[0];
    
    superpixel_labels = superpixel_labels.reshape(0, 1);
    red_im = red_im.reshape(0, 1);
    green_im = green_im.reshape(0, 1);
    blue_im = blue_im.reshape(0, 1);
    
    cv::Mat output = cv::Mat::zeros(4, superpixel_num, CV_32F);
    for (int i = 0; i < (width*height); i++) {
        
        output.at<float>(0, superpixel_labels.at<int>(0, i)) += (float)red_im.at<unsigned char>(0, i);
        output.at<float>(1, superpixel_labels.at<int>(0, i)) += (float)green_im.at<unsigned char>(0, i);
        output.at<float>(2, superpixel_labels.at<int>(0, i)) += (float)blue_im.at<unsigned char>(0, i);
        
        //Counter
        output.at<float>(3, superpixel_labels.at<int>(0, i))++;
    }
    
    
    //Get mean of all the RGB values in the output array:
    //First three rows of output must be divided by last row of output
    
    //Copy 3 rows into one Mat, one Row into the other, and then do an elementwise divide
    cv::Rect roi_red = cv::Rect(0, 0, superpixel_num, 1);
    cv::Rect roi_green = cv::Rect(0, 1, superpixel_num, 1);
    cv::Rect roi_blue = cv::Rect(0, 2, superpixel_num, 1);
    cv::Rect roi_counter = cv::Rect(0, 3, superpixel_num, 1);
    
    cv::Mat r_mean = output(roi_red);
    cv::Mat g_mean = output(roi_green);
    cv::Mat b_mean = output(roi_blue);
    cv::Mat rgb_counter = output(roi_counter);
    
    r_mean = r_mean.mul(1 / rgb_counter);
    g_mean = g_mean.mul(1 / rgb_counter);
    b_mean = b_mean.mul(1 / rgb_counter);
    
    cv::Mat red_im_avg = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat blue_im_avg = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat green_im_avg = cv::Mat::zeros(height, width, CV_32F);
    
    
    int position_label;
    //Populate all of the superpixel segments with the average RGB value
    for (int i = 0; i < height; i++) {
        
        for (int j = 0; j < width; j++) {
            
            //Find label at pixel position
            position_label = labels.at<int>(i, j);
            
            //Go to label position on rgb_mean and set red, blue, and green
            red_im_avg.at<float>(i, j) = r_mean.at<float>(0, position_label);
            green_im_avg.at<float>(i, j) = g_mean.at<float>(0, position_label);
            blue_im_avg.at<float>(i, j) = b_mean.at<float>(0, position_label);
            
        }
        
    }
    
    red_im_avg.convertTo(red_im_avg, CV_8U);
    green_im_avg.convertTo(green_im_avg, CV_8U);
    blue_im_avg.convertTo(blue_im_avg, CV_8U);
    
    vector<cv::Mat> final_channels;
    final_channels.push_back(red_im_avg);
    final_channels.push_back(green_im_avg);
    final_channels.push_back(blue_im_avg);
    
    cv::Mat final_img;
    merge(final_channels, final_img); //final image is actually BGR. Need to fix to clarify later
    
    
    //****************************************************************
    //YCbCr filter for the superpixels
    
    //Convert RGB image to YCrCb:
    cv::Mat ycrcb_img;
    cvtColor(final_img, ycrcb_img, cv::COLOR_BGR2YCrCb); //look at types of images
    
    //Define threshold for y:
    int channel1Min = 0;
    int channel1Max = 255;
    
    //Define threshold for cr:
    int channel2Min = 0;
    int channel2Max = 255;
    
    //Define threshold for cb:
    int channel3Min = 0;
    int channel3Max = 141;
    
    
    //Create binary mask based on thresholds:
    cv::Mat y_im, cr_im, cb_im;
    
    //Mat bw_img = cv::Mat::ones(height, width, CV_32F);
    cv::Mat bw_img = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat bw_img_inv = cv::Mat::ones(height, width, CV_32F);
    
    //Change to be way more efficient later
    vector<cv::Mat> ycrcb_channels;
    split(ycrcb_img, ycrcb_channels);
    
    y_im = ycrcb_channels[0];
    cr_im = ycrcb_channels[1];
    cb_im = ycrcb_channels[2];
    
    //int debug1, debug2, debug3;
    
    //Loop through and perform thresholding
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            
            //threshold
            if (((int)y_im.at<unsigned char>(i, j) >= channel1Min) && ((int)y_im.at<unsigned char>(i, j) <= channel1Max) && ((int)cr_im.at<unsigned char>(i, j) >= channel2Min) && ((int)cr_im.at<unsigned char>(i, j) <= channel2Max) && ((int)cb_im.at<unsigned char>(i, j) >= channel3Min) && ((int)cb_im.at<unsigned char>(i, j) <= channel3Max)) {
                
                bw_img.at<float>(i, j) = 1;
                bw_img_inv.at<float>(i, j) = 0;
            }
        }
    }
    
    
    //convert to unsigned char
    bw_img_inv.convertTo(bw_img_inv, CV_8U);
    
    
    /* COLOR NORMALIZATION */
    
    //Combine mask with image to get blue background
    cv::Mat mask_img;
    int norm_height, norm_width;
    
    bw_img_inv.convertTo(bw_img_inv, CV_8U);
    img.copyTo(mask_img, bw_img_inv);
    
    norm_width = mask_img.size().width;
    norm_height = mask_img.size().height;
    
    //Extract black values from blue background, make channel arrays
    std::list<int> r_channel, g_channel, b_channel;
    cv::Mat test_r, test_g, test_b;
    std::vector<cv::Mat> test_channels;
    
    split(mask_img, test_channels); //BGR
    test_b = test_channels[2];
    test_g = test_channels[1];
    test_r = test_channels[0];
    
    for (int i = 0; i < norm_height; i++) {
        for (int j = 0; j < norm_width; j++) {
            
            if ( (((float)test_b.at<unsigned char>(i, j) == 0) && ((float)test_g.at<unsigned char>(i, j) == 0) && ((float)test_r.at<unsigned char>(i, j) == 0)) == 0 ){
                
                b_channel.push_back((float)test_b.at<unsigned char>(i, j));
                g_channel.push_back((float)test_g.at<unsigned char>(i, j));
                r_channel.push_back((float)test_r.at<unsigned char>(i, j));
            }
        }
    }
    
    //Gamma decode array
    //value_out = A * value_in ^gamma where A is usually 1 and value in is [0, 1]
    
    //Find BGR mean
    double r_sum = 0, g_sum = 0, b_sum = 0;
    double r_output, g_output, b_output;
    float debug = 0;
    
    std::list<int>::iterator g_it, r_it;
    
    g_it = g_channel.begin();
    r_it = r_channel.begin();
    
    for (std::list<int>::iterator b_it = b_channel.begin(); b_it != b_channel.end(); ++b_it) {
        
        b_output = pow( (*b_it / COLOR_MAX), GAMMA_DECODE);
        b_sum += b_output;
        g_output = pow((*g_it / COLOR_MAX), GAMMA_DECODE);
        g_sum += g_output;
        r_output = pow((*r_it / COLOR_MAX), GAMMA_DECODE);
        r_sum += r_output;
        
        g_it = next(g_it);
        r_it = next(r_it);
        
    }
    
    //Create factors to adjust image
    double b_mean_norm, g_mean_norm, r_mean_norm;
    b_mean_norm = (b_sum / b_channel.size()) * COLOR_MAX;
    g_mean_norm = (g_sum / g_channel.size()) * COLOR_MAX;
    r_mean_norm = (r_sum / r_channel.size()) * COLOR_MAX;
    
    double r_factor, g_factor, b_factor;
    b_factor = TEMPLATE_B / b_mean_norm;
    g_factor = TEMPLATE_G / g_mean_norm;
    r_factor = TEMPLATE_R / r_mean_norm;
    
    
    //Gamma decode image
    img.convertTo(img, CV_64FC3);
    cv::Mat new_img(height, width, CV_64FC3);
    new_img = img / COLOR_MAX;
    cv::pow(new_img, GAMMA_DECODE, new_img);
    
    
    //Multiply channels by designated factors
    cv::Mat r_mat, g_mat, b_mat, norm_img;
    std::vector<cv::Mat> img_channels, norm_channels;
    
    split(new_img, img_channels); //BGR
    b_mat = img_channels[2];
    g_mat = img_channels[1];
    r_mat = img_channels[0];
    
    b_mat = b_mat * b_factor;
    g_mat = g_mat * g_factor;
    r_mat = r_mat * r_factor;
    
    norm_channels.push_back(b_mat);
    norm_channels.push_back(g_mat);
    norm_channels.push_back(r_mat);
    merge(norm_channels, norm_img);
    
    //Gamma encode image
    cv::pow(norm_img, GAMMA_ENCODE, norm_img);
    norm_img = norm_img * COLOR_MAX;
    norm_img.convertTo(norm_img, CV_8U);
    
    split(norm_img, norm_channels);
    //Remove this and replace with normalized color image
    cv::Mat red_saddle_im, blue_saddle_im, green_saddle_im;
    red_saddle_im   = norm_channels[2];
    green_saddle_im = norm_channels[1];
    blue_saddle_im  = norm_channels[0];
    
    //locate centroid of image
    cv::Moments m = moments(bw_img, true);
    
    //pull out centroid coordinates
    int center_x = (int)m.m10 / m.m00;
    int center_y = (int)m.m01 / m.m00;
    
    //deal with multiple centroids
    //find outline of the image
    cv::Mat outline;
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2), cv::Point(-1, -1));
    morphologyEx(bw_img, outline, 4, element); //4 is Morphological Gradient
    
    //slope heatmap
    double delta_x, delta_y, slope;
    
    
    //Loop through and perform thresholding
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            
            if (outline.at<float>(i, j) == 1) {
                
                //Find slope of the points
                delta_x = (double)j - center_x;
                delta_y = (double)i - center_y;
                
                slope = delta_y / delta_x;
                //cout << slope << endl;
                
                if (!((slope < 0.8) && (slope > -0.8))) {
                    outline.at<float>(i, j) = 0;
                }
            }
        }
    }
    
    
    int row, col;
    int state = 0; //initialize state machine
    
    //Traverse through image to fill in mask
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            
            switch (state) {
                    
                case 0: //STATE 0
                    
                    if (outline.at<float>(i, j) == 1) {
                        state = 1;
                    }
                    else {
                        state = 0;
                    }
                    
                    break;
                    
                case 1: //STATE 1
                    
                    if (outline.at<float>(i, j) == 1) { //takes care of thickness
                        state = 1;
                    }
                    else {
                        outline.at<float>(i, j) = 1;
                        state = 2;
                    }
                    break;
                    
                case 2: //STATE 2
                    
                    if (outline.at<float>(i, j) == 0) {
                        outline.at<float>(i, j) = 1;
                        state = 2;
                    }
                    else {
                        state = 3;
                    }
                    
                    if (((j == width - 1) || (j == 0)) && (outline.at<float>(i, j) == 1)) {
                        //clear the entire row
                        for (int k = 0; k < width; k++) {
                            outline.at<float>(i, k) = 0;
                            
                        }
                        state = 0;
                    }
                    
                    break;
                    
                case 3: //STATE 3
                    
                    if (outline.at<float>(i, j) == 1) {
                        state = 3;
                    }
                    else {
                        state = 4;
                    }
                    
                    if (((j == width - 1) || (j == 0)) && (outline.at<float>(i, j) == 1)) {
                        //clear the entire row
                        for (int k = 0; k < width; k++) {
                            outline.at<float>(i, k) = 0;
                            
                        }
                        state = 0;
                    }
                    
                    break;
                    
                case 4: //STATE 4
                    //handles either the end or outline overlaps
                    if (outline.at<float>(i, j) == 1) {
                        row = i;
                        col = j;
                        while (1) {
                            if (outline.at<float>(row, col) == 0) {
                                outline.at<float>(row, col) = 1;
                                col--;
                            }
                            else {
                                break;
                            }
                        }
                    }
                    
                    state = 4; //persistent state until end
                    
                    if (j == width - 1) {
                        state = 0;
                    }
                    
                    break;
                    
            }
            
        }
    }
    
    //Apply saddle region mask to peanut, extract RGB info and find average RBG values
    double red_saddle_sum, green_saddle_sum, blue_saddle_sum, px_num;
    
    red_saddle_sum = 0;
    green_saddle_sum = 0;
    blue_saddle_sum = 0;
    px_num = 0;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            
            if (outline.at<float>(i, j) == 1) {
                
                red_saddle_sum += (double)red_saddle_im.at<unsigned char>(i, j);
                green_saddle_sum += (double)green_saddle_im.at<unsigned char>(i, j);
                blue_saddle_sum += (double)blue_saddle_im.at<unsigned char>(i, j);
                px_num++;
                
            }
        }
    }
    
    double red_saddle_avg, green_saddle_avg, blue_saddle_avg;
    red_saddle_avg = red_saddle_sum / px_num;
    green_saddle_avg = green_saddle_sum / px_num;
    blue_saddle_avg = blue_saddle_sum / px_num;
    
    std::vector<double> results = *new std::vector<double>(3);
    //double result_avg[3];
    //result_avg[0] = (int)round(red_saddle_avg);
    //result_avg[1] = (int)round(green_saddle_avg);
    //result_avg[2] = (int)round(blue_saddle_avg);
    results[0] = (int)round(red_saddle_avg);
    results[1] = (int)round(green_saddle_avg);
    results[2] = (int)round(blue_saddle_avg);
    
    //results[0] = result_avg[0];
    return results;
}

//classifies data
std::vector<float> classify(std::vector<double> avg, std::vector<double> r_avg_arr, std::vector<double> g_avg_arr, std::vector<double> b_avg_arr, std::vector<double> label_arr){
    
    using namespace std;
    using namespace nanoflann;
    
    size_t size = r_avg_arr.size();
    std::vector<double> r_avg = r_avg_arr;
    std::vector<double> g_avg = g_avg_arr;
    std::vector<double> b_avg = b_avg_arr;
    std::vector<double> label = label_arr;
    
    
    //Generate point cloud for KDTree
    PointCloud<double> cloud;
    const int N = r_avg.size();
    createPointCloud(cloud, N, r_avg, g_avg, b_avg);
    
    size_t k = 5;
    
    //Initialize K-NN model
    // construct a kd-tree index:
    typedef KDTreeSingleIndexAdaptor<
    L2_Adaptor<double, PointCloud<double> >,
    PointCloud<double>,
    3> my_kd_tree;
    dump_mem_usage();
    
    my_kd_tree   index(k, cloud, KDTreeSingleIndexAdaptorParams(2));
    index.buildIndex();
    dump_mem_usage();
    
    std::vector<size_t>   ret_index(k+1);
    std::vector<double> out_dist_sqr(k+1);
    double query_pt[3];
    int neighbor[5];
    double class0, class1, class2, class3, class4;
    
    //Create membership values for the training points:
    for (int i = 0; i < N; i++)
    {
        //std::cout << i << endl;
        class0 = class1 = class2 = class3 = class4 = 0;
        query_pt[0] = cloud.pts[i].x; //r_avg[i];
        query_pt[1] = cloud.pts[i].y; //g_avg[i];
        query_pt[2] = cloud.pts[i].z; //b_avg[i];
        
        //Find the k-nearest neighbors of the training point
        index.knnSearch(&query_pt[0], k+1, &ret_index[0], &out_dist_sqr[0]);
        
        //Find labels for nearest neighbors
        neighbor[0] = label[ret_index[1]];
        neighbor[1] = label[ret_index[2]];
        neighbor[2] = label[ret_index[3]];
        neighbor[3] = label[ret_index[4]];
        neighbor[4] = label[ret_index[5]];
        
        //Find sums of classes present
        for (int i = 0; i < 5; i++) {
            
            switch (neighbor[i]) {
                case 0:
                    class0 += 1;
                    break;
                case 1:
                    class1 += 1;
                    break;
                case 2:
                    class2 += 1;
                    break;
                case 3:
                    class3 += 1;
                    break;
                case 4:
                    class4 += 1;
                    break;
                default:
                    break;
            }
            
        }
        
        //Fill out membership values for datapoint
        cloud.pts[i].member[0] = 0.49 * (class0 / 5);
        cloud.pts[i].member[1] = 0.49 * (class1 / 5);
        cloud.pts[i].member[2] = 0.49 * (class2 / 5);
        cloud.pts[i].member[3] = 0.49 * (class3 / 5);
        cloud.pts[i].member[4] = 0.49 * (class4 / 5);
        
        //Add membership for value that is in correct class
        if (label[i] == 0) {
            cloud.pts[i].member[0] += (0.51 - (0.49 / k));
        }
        else if (label[i] == 1) {
            cloud.pts[i].member[1] += (0.51 - (0.49 / k));
        }
        else if (label[i] == 2) {
            cloud.pts[i].member[2] += (0.51 - (0.49 / k));
        }
        else if (label[i] == 3) {
            cloud.pts[i].member[3] += (0.51 - (0.49 / k));
        }
        else { //label[i] == 4
            cloud.pts[i].member[4] += (0.51 - (0.49 / k));
        }
        
    }
    
    //Get query point:
    //double* new_query_pt = (double*)avg;
    size_t new_size = avg.size();
    std::vector<double> input( new_size );
    input = avg;
    
    double new_query_pt[3];
    new_query_pt[0] = input[0];
    new_query_pt[1] = input[1];
    new_query_pt[2] = input[2];
    
    
    double temp0, temp1, temp2, temp3, temp4, denomSum;
    double finalMember[5];
    float m = 2.0; //fuzzifier
    index.knnSearch(&new_query_pt[0], k, &ret_index[0], &out_dist_sqr[0]);
    
    float eta1 = 0.1;
    float eta2 = 0.5;
    
    if( (sqrt(out_dist_sqr[0]) - (eta1/eta2)) > 0 ){
        temp0 = sqrt(out_dist_sqr[0]) - (eta1/eta2);
    }else{
        temp0 = 0;
    }
    if( (sqrt(out_dist_sqr[1]) - (eta1/eta2)) > 0 ){
        temp1 = sqrt(out_dist_sqr[1]) - (eta1/eta2);
    }else{
        temp1 = 0;
    }
    if( (sqrt(out_dist_sqr[2]) - (eta1/eta2)) > 0 ){
        temp2 = sqrt(out_dist_sqr[2]) - (eta1/eta2);
    }else{
        temp2 = 0;
    }
    if( (sqrt(out_dist_sqr[3]) - (eta1/eta2)) > 0 ){
        temp3 = sqrt(out_dist_sqr[3]) - (eta1/eta2);
    }else{
        temp3 = 0;
    }
    if( (sqrt(out_dist_sqr[4]) - (eta1/eta2)) > 0 ){
        temp4 = sqrt(out_dist_sqr[4]) - (eta1/eta2);
    }else{
        temp4 = 0;
    }
    
    float value = 2/(m-1);
    
    temp0 = 1.0 / (1.0 + pow(temp0, (2/(m - 1))));
    temp1 = 1.0 / (1.0 + pow(temp1, (2/(m - 1))));
    temp2 = 1.0 / (1.0 + pow(temp2, (2/(m - 1))));
    temp3 = 1.0 / (1.0 + pow(temp3, (2/(m - 1))));
    temp4 = 1.0 / (1.0 + pow(temp4, (2/(m - 1))));
    
    double weights[5] = {temp0, temp1, temp2, temp3, temp4};
    
    for (int i=0; i<5; i++) {
        for (int j=0; j<5; j++) {
            finalMember[i] = finalMember[i] + (weights[j] * cloud.pts[ret_index[j]].member[i]);
        }
    }
    
    std::vector<float> results = *new std::vector<float>(5);
    float finalClass[5];
    
    finalClass[0] = abs(finalMember[0]);
    finalClass[1] = abs(finalMember[1]);
    finalClass[2] = abs(finalMember[2]);
    finalClass[3] = abs(finalMember[3]);
    finalClass[4] = abs(finalMember[4]);
    
    for(int i = 0; i < 5; ++i){
        results[i] = finalClass[i];
    }
    return results;
}


/*+ (UIImage *)makeGray:(UIImage *)image {
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
 
 }*/

+ (NSString *)classifyPeanut:(UIImage *)image {
    
    try{
    //transform UIImage to cv::mat
    cv::Mat imageMat;
    UIImageToMat(image, imageMat);
    
    std::vector<double> avgRGB = preprocess(&imageMat);
    
    std::vector<double> red_avg, blue_avg, green_avg, label;
    std::ifstream iffy;
    iffy.open("Documents/r_avg.txt");
    if(iffy.is_open()){
        std::cout << "we here girl\n\n";
    }
    else{
        std::cout << " sad face \n\n";
    }
    iffy.close();
    
    red_avg = {184,51,161,67,93,117,124,47,77,157,44,130,142,27,40,199,115,164,76,47,60,126,36,76,113,214,183,160,50,78,49,67,40,155,42,47,62,169,160,152,61,68,55,19,94,170,132,44,91,222,41,54,40,28,148,88,50,162,40,60,162,72,49,50,25,57,54,87,47,29,181,155,147,37,190,39,65,47,168,57,44,81,169,28,170,194,161,218,59,83,34,73,110,74,179,55,166,145,174,55,42,68,175,54,158,66,174,123,129,77,42,79,52,82,68,55,131,149,186,47,143,71,36,128,55,168,40,50,141,47,168,63,58,67,50,73,162,153,172,113,38,30,18,180,44,147,175,177,148,67,44,125,126,41,84,34,200,33,44,142,180,171,75,69,45,69,120,41,182,130,175,52,98,60,57,31,55,40,152,196,182,129,87,41,33,174,163,44,160,21,109,134,37,45,49,166,178,105,54,149,37,224,212,164,61,192,225,193,185,79,77,131,140,65,48,33,51,167,45,34,49,27,29,153,131,99,166,180,178,39,33,148,66,156,82,197,59,121,51,44,39,49,45,119,72,98,40,71,35,67,52,61,141,64,109,46,180,34,40,37,151,153,34,49,165,144,47,37,193,194,54,25,145,53,130,31,61,47,157,39,45,63,186,84,45,69,143,219,67,39,110,178,171,80,200,40,177,67,174,60,163,48,148,49,198,95,48,46,156,20,145,172,148,52,169,167,18,151,32,181,69,99,39,76,47,19,29,114,46,92,124,152,228,179,181,181,46,178,59,41,92,42,202,44,67,41,152,146,116,133,47,152,132,35,75,83,58,91,183,74,175,45,197,54,39,136,142,22,70,83,43,60,27,47,30,36,183,77,182,38,176,225,48,37,58,39,122,148,18,113,109,88,77,111,45,219,147,100,66,199,58,73,50,59,50,132,43,138,74,31,162,111,37,47,106,53,151,181,39,92,133,64,39,71,164,65,133,154,67,64,60,64,53,35,58,62,59,98,226,103,99,30,44,52,81,40,171,145,71,75,57,52,71,41,103,118,155,229,42,30,166,120,87,35,165,64,61,61,73,49,47,181,31,61,42,38,112,153,155,116,62,31,164};
    
    green_avg = {156,51,138,70,72,94,107,42,66,121,37,103,127,26,37,175,104,99,57,35,58,86,38,67,109,185,149,142,41,65,47,45,37,114,43,49,55,159,152,132,52,54,32,22,83,161,106,47,85,168,32,50,40,21,123,72,51,142,43,53,115,71,45,48,24,57,54,82,47,33,146,136,95,36,131,42,64,45,115,51,42,53,152,28,125,171,132,172,55,71,28,58,83,53,136,44,154,109,153,47,46,56,104,43,136,42,167,81,84,55,42,65,53,61,54,54,102,134,145,47,127,77,38,102,47,145,41,50,111,49,134,65,50,59,53,67,128,146,145,86,40,27,22,159,44,119,150,147,113,42,47,89,106,32,54,38,148,27,23,126,161,158,65,61,46,54,93,34,151,127,158,50,66,57,56,33,33,37,127,166,108,97,61,44,28,143,142,43,119,24,78,135,39,45,48,150,80,87,50,129,33,158,184,151,47,174,161,155,137,52,67,92,108,42,40,28,38,156,46,39,43,32,26,113,119,73,148,155,158,41,27,126,53,135,78,169,54,87,47,37,36,39,45,80,72,72,42,56,32,56,47,59,102,51,94,48,166,37,38,19,125,132,37,50,148,134,36,40,147,156,50,26,134,53,107,35,61,36,128,41,50,58,167,78,38,66,109,167,64,43,73,148,137,67,137,39,157,56,163,58,145,43,100,47,169,76,46,37,119,23,131,152,134,44,151,126,17,121,33,135,57,76,40,104,49,21,28,89,45,73,91,137,154,143,159,157,42,142,45,38,78,39,178,38,43,33,119,126,90,104,48,136,101,31,45,77,52,84,166,59,150,43,131,53,34,109,141,25,61,53,45,57,31,49,31,32,163,50,162,41,97,177,39,38,46,36,94,129,21,87,73,62,69,85,41,170,100,102,54,171,45,67,53,45,38,132,44,117,63,28,110,89,38,48,74,54,107,154,34,74,94,59,39,69,153,55,134,133,53,43,59,52,54,38,49,61,41,65,208,82,81,34,46,44,65,44,141,93,50,77,63,45,68,40,75,94,145,207,44,27,118,103,84,35,150,62,57,60,62,51,45,171,29,59,38,30,85,115,142,91,60,34,141};
    
    blue_avg = {102,43,84,54,36,47,71,31,47,72,22,56,70,17,28,110,81,42,35,21,45,31,34,47,90,97,82,77,32,51,36,27,26,50,35,43,47,97,104,71,38,35,16,18,61,103,55,40,68,112,22,41,41,18,75,46,41,80,36,41,49,58,36,37,20,41,42,62,41,29,72,76,30,27,53,35,52,37,51,37,35,28,81,23,56,111,63,74,42,44,20,45,48,37,58,32,88,45,93,37,43,48,30,28,92,24,129,36,30,34,36,40,56,37,44,47,58,97,69,39,78,80,38,52,33,82,39,37,70,43,87,58,39,42,39,47,57,94,85,43,40,19,18,85,38,62,87,79,61,25,46,49,58,24,31,32,92,17,18,67,105,98,55,51,38,38,49,26,81,104,104,46,37,41,41,28,27,28,66,82,27,49,41,37,18,81,96,37,49,21,35,107,33,35,40,85,21,54,40,74,26,39,116,90,40,110,100,71,69,28,50,39,67,24,31,20,26,93,41,40,31,29,21,52,72,48,82,92,96,32,20,70,29,86,60,118,38,40,35,25,35,40,37,31,58,35,36,42,31,50,33,49,66,35,59,37,99,31,38,12,73,73,31,40,86,86,29,33,80,75,47,21,128,49,59,30,47,27,59,34,41,44,105,52,34,49,55,112,44,40,35,81,96,53,49,33,108,44,110,44,83,39,38,35,106,54,46,26,66,20,77,79,78,35,89,58,14,62,28,59,40,58,35,132,41,18,23,54,37,45,44,81,49,74,95,93,32,64,41,40,55,30,102,27,23,20,54,78,44,56,43,81,51,21,24,63,37,85,94,39,87,35,52,51,25,54,106,20,53,28,38,46,29,45,30,24,100,33,97,34,28,76,37,32,32,29,43,76,19,56,35,40,61,55,29,74,36,94,35,118,29,52,48,32,27,101,38,68,51,18,68,49,33,35,33,38,45,88,24,48,39,42,39,56,96,47,109,72,44,19,49,45,44,32,46,42,27,34,138,46,60,31,36,34,33,38,69,26,33,76,66,34,53,31,44,50,94,108,35,19,57,62,66,30,85,55,44,51,35,51,32,101,22,47,28,20,42,47,91,50,56,29,72};
    
    label = {3,0,3,0,1,2,3,1,1,1,0,2,3,0,0,3,1,1,1,0,1,2,0,1,1,2,3,3,1,0,1,0,1,2,0,0,1,3,3,3,1,1,0,0,1,3,2,0,1,4,1,1,1,0,2,1,0,2,0,1,2,1,1,0,0,1,1,0,1,0,3,3,2,0,3,0,0,0,2,1,1,1,2,0,2,3,2,3,1,1,1,1,1,0,3,1,2,2,3,1,0,1,2,1,1,0,3,1,2,1,0,1,1,1,1,1,1,1,3,1,3,0,0,2,0,3,0,0,1,0,3,1,1,1,0,1,2,3,3,2,0,0,0,3,0,3,3,2,2,1,1,1,2,1,1,0,3,0,1,3,3,3,1,1,0,1,2,0,3,4,3,0,1,1,0,1,1,1,2,1,2,2,1,0,0,2,3,1,2,0,1,2,0,1,1,3,1,2,1,2,0,2,3,3,1,3,3,3,2,1,1,2,2,1,1,0,1,3,0,0,1,0,1,2,3,1,2,3,3,0,1,3,1,3,0,3,1,2,0,1,1,1,1,2,0,1,1,1,0,1,1,0,1,1,1,1,3,1,1,1,2,3,0,0,3,3,1,0,2,3,1,0,1,0,2,0,1,1,2,0,0,1,3,1,1,1,2,4,0,0,1,2,3,1,3,0,3,1,3,1,3,0,2,1,1,1,1,1,2,0,3,3,3,1,3,2,0,3,1,2,1,1,0,3,0,0,0,2,1,2,2,3,2,3,3,3,0,3,1,0,1,1,3,1,0,0,3,3,2,1,0,3,2,0,0,1,1,1,3,1,3,1,2,1,1,2,3,0,1,1,0,0,0,0,0,0,3,0,3,0,1,3,1,0,1,1,3,3,0,0,1,0,1,1,0,2,1,2,1,1,0,0,0,1,1,3,1,3,1,0,2,2,0,0,2,0,1,3,0,1,2,1,0,1,3,1,3,2,1,1,0,1,1,1,1,1,1,0,3,1,1,0,0,0,3,0,3,2,1,1,0,1,0,0,1,1,2,3,1,0,2,1,0,0,3,0,0,0,0,1,0,3,0,1,1,1,2,2,2,2,1,0,2};
    
    std::vector<float> classWeights = classify(avgRGB, red_avg, green_avg, blue_avg, label);
    
    std::string str;
    double max = 0.0;
    int maxIndex = -1;
    for(int i = 0; i < classWeights.size(); i++){
        if(classWeights[i] > max){
            max = classWeights[i];
            maxIndex = i;
        }
    }
    
    
    str = "This peanut is classified as: ";
    if(max < 0.009){
        
        str += "Unable to Classify";
        
    }else{
        switch(maxIndex){
                
            case 0:
                str += "Black";
                break;
            case 1:
                str += "Brown";
                break;
            case 2:
                str += "Orange";
                break;
            case 3:
                str += "Yellow";
                break;
            case 4:
                str += "White";
                break;
                
        }
    }
    
    return [NSString stringWithUTF8String:str.c_str()];
    }
    catch (const std::overflow_error& e) {
        std::cout << "Overflow error.\n";
        std::string err = "Error: not detected as a peanut.";
        return [NSString stringWithUTF8String:err.c_str()];
    }
    catch (const std::runtime_error& e) {
        std::cout << "Runtime error.\n";
        std::string err = "Error: not detected as a peanut.";
        return [NSString stringWithUTF8String:err.c_str()];
    }
    catch (const std::exception& e) {
        std::cout << "Logic error.\n";
        std::string err = "Error: not detected as a peanut.";
        return [NSString stringWithUTF8String:err.c_str()];
    }
    catch (...) {
        std::cout << "Generic error.\n";
        std::string err = "Error: not detected as a peanut.";
        return [NSString stringWithUTF8String:err.c_str()];
    }
}


/*
 NSString * stringFromArray = NULL;
 NSMutableArray * array = [[NSMutableArray alloc] initWithCapacity: 43];
 if(array)
 {
 NSInteger count = 0;
 
 while( count++ < 43 )
 {
 [array addObject: [NSString stringWithFormat: @"%f", c_array[count]]];
 }
 
 stringFromArray = [array componentsJoinedByString:@","];
 [array release];
 }
 */

@end
