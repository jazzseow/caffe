#include <iostream>
#include "caffe/caffe.hpp"
#include "ristretto/quantization.hpp"
#include "caffe/util/bbox_util.hpp"
using caffe::Caffe;
using caffe::Net;
int main(){
  string model = "models/VGGNet/VOC0712/SSD_300x300_ft/train.prototxt";
  string weights = "models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel";
  Net<float>* net_test = new Net<float>(model, caffe::TRAIN);
  net_test->CopyTrainedLayersFrom(weights);
  vector<float> max_in_, max_params_, max_out_;
  vector<string> layer_names_;
  
  for (int i=0;i<3;i++){
    net_test->Forward();
    net_test->RangeInLayers(&layer_names_, &max_in_, &max_out_,
                  &max_params_);
  }
   for (int i=0;i<layer_names_.size();i++){
    std::cout<<layer_names_[i]<<std::endl;
    std::cout<<max_in_[i]<<std::endl;
    std::cout<<max_out_[i]<<std::endl;
    std::cout<<max_params_[i]<<std::endl;
  }
  return 0;
}
