#pragma once
#include <array>
#include <span>
#include <cmath>
#include <stdexcept>
namespace photo_rays {
using V=std::array<double,3>;using M=std::array<double,9>;
struct Line {V center,direction;};
// Frame-camera path 143A6E840: closest intersection of world-space rays.
// The default path returns zero for a singular system. The specialized
// 142EB24B0 path supplies a condition limit and falls back to the mean center.
inline V intersect(std::span<const Line>lines,double condition=0){
 if(lines.empty())throw std::runtime_error("empty ray intersection");M a{};V b{},mean{};
 for(auto line:lines){auto c=line.center,d=line.direction;double length=std::sqrt(d[1]*d[1]+d[0]*d[0]+d[2]*d[2]),scale=length>=1e-20?1/length:0;for(auto&v:d)v*=scale;
  double xx=1-d[0]*d[0],yy=1-d[1]*d[1],zz=1-d[2]*d[2],xy=d[1]*d[0],xz=d[2]*d[0],yz=d[2]*d[1];
  a[0]+=xx;a[1]-=xy;a[2]-=xz;a[3]-=xy;a[4]+=yy;a[5]-=yz;a[6]-=xz;a[7]-=yz;a[8]+=zz;
  b[0]+=xx*c[0]-c[1]*xy-c[2]*xz;b[1]+=c[1]*yy-xy*c[0]-c[2]*yz;b[2]+=-d[0]*d[2]*c[0]-c[1]*yz+c[2]*zz;
  for(unsigned k=0;k<3;++k)mean[k]+=c[k];
 }
 double A=a[4]*a[8]-a[5]*a[7],G=a[3]*a[7]-a[6]*a[4],det=a[0]*A-(a[3]*a[8]-a[6]*a[5])*a[1]+a[2]*G;
 if(det!=0){double r=1/det;M inv{A*r,(a[2]*a[7]-a[1]*a[8])*r,(a[1]*a[5]-a[2]*a[4])*r,(a[6]*a[5]-a[3]*a[8])*r,(a[0]*a[8]-a[2]*a[6])*r,(a[2]*a[3]-a[0]*a[5])*r,G*r,(a[1]*a[6]-a[0]*a[7])*r,(a[0]*a[4]-a[1]*a[3])*r};double na=0,nb=0;for(double v:a)na+=v*v;for(double v:inv)nb+=v*v;if(condition==0||na*nb<condition*condition)return {b[1]*inv[1]+b[0]*inv[0]+b[2]*inv[2],b[1]*inv[4]+b[0]*inv[3]+b[2]*inv[5],b[1]*inv[7]+b[0]*inv[6]+b[2]*inv[8]};}
 if(condition==0)return {}; double scale=1/double(lines.size());for(auto&v:mean)v*=scale;return mean;
}
}
