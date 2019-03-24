//
//  OpenCVWrapper.h
//  SeniorProject
//
//  Created by Ross Spencer on 2/10/19.
//  Copyright © 2019 Ross Spencer. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

//objective-C declarations
NS_ASSUME_NONNULL_BEGIN

@interface OpenCVWrapper : NSObject

//gets openCV version number
+ (NSString *)openCVVersionString;

//gets greyscale version of image
+ (UIImage *) makeGray: (UIImage *) image;

//#if defined(__cplusplus)
//extern "C" {
//#endif
+ (NSString *)classifyPeanut:(UIImage *)image ;
//#if defined(__cplusplus)
//}
//#endif

@end

NS_ASSUME_NONNULL_END
