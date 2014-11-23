#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <opencv/ml.h>

#define voodoo ((Members*) pimpl)

struct Members {
  bool first_dataset = true;
  cv::NormalBayesClassifier bayes_classifier;
};

/// Constructor
SkinModel::SkinModel()
{
  Members* members = new Members();
  pimpl = (SkinModelPimpl*) members;
}

/// Destructor
SkinModel::~SkinModel() 
{
}

/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
  cv::Mat trainData(img.rows*img.cols, 3, CV_32F);
  cv::Mat responses(img.rows*img.cols, 1, CV_32S);

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      cv::Vec3b bgr = img(row, col);
      trainData.at<float>((row*col)+col, 0) = bgr[0];
      trainData.at<float>((row*col)+col, 1) = bgr[1];
      trainData.at<float>((row*col)+col, 2) = bgr[2];
      responses.at<int>((row*col)+col, 0) = ((int)mask(row, col) == 255);
      //std::cout << "[" << (int)bgr[0] << "," << (int)bgr[1] << "," << (int)bgr[2] << "]" << "<--->" << ((int)mask(row,col) == 255) << std::endl;
    }
  }
  // train the bayes classifier
  voodoo->bayes_classifier.train(trainData, responses, cv::Mat(), cv::Mat(), !voodoo->first_dataset);
  voodoo->first_dataset = false;
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
	//--- IMPLEMENT THIS ---//
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
  cv::Mat1b skin(img.rows, img.cols);

  //cv::Mat samples(img.rows*img.cols, 3, CV_32F);
  cv::Mat samples(1, 3, CV_32F);
  cv::Mat responses(1, 1, CV_32S);

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      cv::Vec3b bgr = img(row, col);
      samples.at<float>(0, 0) = bgr[0];
      samples.at<float>(0, 1) = bgr[1];
      samples.at<float>(0, 2) = bgr[2];
      skin.at<char>(row, col) = voodoo->bayes_classifier.predict(samples, &responses);
      //std::cout << "skin is " << skin(row, col) << ", responses is " << responses.at<float>(0,0) << std::endl;
    }
  }

  return skin;
}

