#pragma once
#include "pair_optimize.hpp"
namespace photo_initial {
inline Matrix frame_intrinsic(unsigned width,unsigned height,double f){return {f,0,double(width)*.5,0,f,double(height)*.5,0,0,1};}
struct RefinedRelative{Projection first,second;Camera camera;std::vector<Pixel> pixels_first,pixels_second;PairSolution optimization;};
// 143A142D0 projects to an orthogonal matrix with SVD before its logarithm.
inline std::array<double,3> rotation_vector(const Matrix&input){auto svd=ms::five::square_svd<3>(input);auto r=multiply(transpose(svd.ut),svd.vt);double x=r[7]-r[5],y=r[2]-r[6],z=r[3]-r[1];double sine=std::sqrt((y*y+x*x+z*z)*.25);double cosine=std::min(1.0,std::max(-1.0,(r[0]+r[4]+r[8]-1)*.5));double angle=std::acos(cosine);
 if(sine<1e-5){if(cosine>0)return {0,0,0};return {std::sqrt(std::max((r[0]+1)*.5,0.0))*angle,std::sqrt(std::max((r[4]+1)*.5,0.0))*angle*(r[1]<0?-1:1),std::sqrt(std::max((r[8]+1)*.5,0.0))*angle*(r[2]<0?-1:1)};}
 double scale=(1/(sine*2))*angle;return {x*scale,y*scale,z*scale};
}
inline PairSolution refine_pair(Projection&first,Projection&second,std::span<const Pixel>a,std::span<const Pixel>b,std::span<const unsigned char>mask,double tolerance=1e-6,unsigned iterations=100){
 if(a.size()!=b.size()||a.size()!=mask.size())throw std::runtime_error("pair refinement extent");auto p=first,q=second;canonicalize(p,q);auto ca=decompose(p),cb=decompose(q);auto rotation=rotation_vector(cb.rotation);std::array<double,6>g{rotation[0],rotation[1],rotation[2],cb.center[0],cb.center[1],cb.center[2]};std::vector<std::array<double,3>>points;std::vector<std::array<double,4>>observations;
 for(unsigned i=0;i<a.size();++i)if(mask[i]){auto x=triangulate_pixels(p,q,a[i],b[i]);if(x[2]==0)throw std::runtime_error("pair inverse depth at infinity");points.push_back({x[0]/x[2],x[1]/x[2],x[3]/x[2]});observations.push_back({a[i][0],a[i][1],b[i][0],b[i][1]});}
 auto result=optimize_pair(g,std::move(points),{ca.intrinsic,cb.intrinsic},observations,tolerance,iterations);auto&out=result.global;first=compose({ca.intrinsic,{1,0,0,0,1,0,0,0,1},{0,0,0}});second=compose({cb.intrinsic,rodrigues({out[0],out[1],out[2]}).r,{out[3],out[4],out[5]}});return result;
}
// Frame branch of 142E8EA70 after essential estimation / pose selection.
inline RefinedRelative refine_relative(Camera camera,std::span<const std::array<double,3>>a,std::span<const std::array<double,3>>b,std::span<unsigned char>mask,const Matrix&ka,const Matrix&kb,double ta,double tb){
 if(a.size()!=b.size()||a.size()!=mask.size())throw std::runtime_error("relative refinement extent");
 RefinedRelative result;result.first=compose({ka,{1,0,0,0,1,0,0,0,1},{0,0,0}});camera.intrinsic=kb;result.second=compose(camera);
 for(unsigned i=0;i<a.size();++i){result.pixels_first.push_back({a[i][0]*ka[0]+a[i][1]*ka[1]+a[i][2]*ka[2],a[i][0]*ka[3]+a[i][1]*ka[4]+a[i][2]*ka[5]});result.pixels_second.push_back({b[i][0]*kb[0]+b[i][1]*kb[1]+b[i][2]*kb[2],b[i][0]*kb[3]+b[i][1]*kb[4]+b[i][2]*kb[5]});}
 result.optimization=refine_pair(result.first,result.second,result.pixels_first,result.pixels_second,mask);filter_pair(fundamental(result.first,result.second),result.pixels_first,result.pixels_second,mask,ta,tb);canonicalize(result.first,result.second);result.camera=decompose(result.second);return result;
}
}
