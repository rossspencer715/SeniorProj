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
    //transform UIImage to cv::mat
    cv::Mat imageMat;
    UIImageToMat(image, imageMat);
    
    std::vector<double> classWeights = preprocess(&imageMat);
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
        str += std::to_string(maxIndex);
    return [NSString stringWithUTF8String:str.c_str()];
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
