#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace photo_initial {
// Frame fallback in 1422907A0: image diagonal / 36x24 mm diagonal, 50 mm.
// Do not use width*50/36: that is equivalent only for a 3:2 image.
inline std::array<std::byte,256> no_exif_calibration(std::uint64_t width,std::uint64_t height){
 if(!width||!height||width>INT32_MAX||height>INT32_MAX)throw std::runtime_error("invalid image dimensions");
 std::array<std::byte,256> c{};
 auto put=[&](unsigned offset,auto value){std::memcpy(c.data()+offset,&value,sizeof(value));};
 put(0,std::uint32_t{1});put(8,width);put(16,height);
 double w=double(width),h=double(height);
 put(24,std::sqrt(w*w+h*h)*50.0/std::sqrt(1872.0));c[72]=std::byte{1};
 // 142347000 constructor, including the valid image/radius bounds.
 put(176,1e9);put(184,(std::numeric_limits<double>::max)());put(192,(std::numeric_limits<double>::max)());
 put(200,-(std::numeric_limits<double>::max)());put(208,-(std::numeric_limits<double>::max)());
 return c;
}
// Pixel tolerance matches the captured 3072x2048 no-EXIF sensor (10.2400004864).
// The mean-dimension generalization still needs non-3:2 capture verification.
// 14234C250 converts it to the angular parameter. 143AB21F0 generates 10 levels.
inline std::array<double,10> thresholds(std::uint64_t width,std::uint64_t height,double focal,double relative_accuracy=double(0.004f)){
 if(!(focal>0)||!std::isfinite(focal)||!(relative_accuracy>0))throw std::runtime_error("invalid ray threshold");
 std::array<double,10> t{};double pixels=(double(width)+double(height))*0.5*relative_accuracy;
 t[0]=(1.0/focal)*pixels;for(unsigned i=1;i<10;++i)t[i]=t[i-1]*0.75;return t;
}
}
