#pragma once
#include "evidence.hpp"
using namespace ms;
using Pixel=std::array<double,2>;using Ray=std::array<double,3>;
struct Observation{unsigned image,track;std::uint64_t auxiliary;float u,v,depth,size;};
static_assert(sizeof(Observation)==32);
template<class T>T field(std::span<const std::byte> b,std::size_t offset){if(offset+sizeof(T)>b.size())throw std::runtime_error("field extent");T v;std::memcpy(&v,b.data()+offset,sizeof v);return v;}
// 142EA6970 preserves per-image order and merge-joins sorted track identifiers.
void common(std::span<const Observation> observations,std::uint64_t first,std::uint64_t second,std::vector<Pixel>& a,std::vector<Pixel>& b){
 std::vector<Observation> left,right;for(auto p:observations){if(p.image==first)left.push_back(p);if(p.image==second)right.push_back(p);}
 auto less=[](auto x,auto y){return x.track<y.track;};
 if(!std::is_sorted(left.begin(),left.end(),less)||!std::is_sorted(right.begin(),right.end(),less))throw std::runtime_error("pair observations are not ordered by track");
 std::size_t j=0;for(auto p:left){while(j<right.size()&&right[j].track<p.track)++j;if(j==right.size())break;if(right[j].track==p.track){a.push_back({p.u,p.v});b.push_back({right[j].u,right[j].v});}}
}
// 14234CDC0 frame branch and 142642F80 inverse distortion, strict double order.
Ray unproject(std::span<const std::byte> c,Pixel p){
 auto type=field<unsigned>(c,0);if(type>1)throw std::runtime_error("only perspective/frame sensors supported");
 // The current dataset has no per-pixel correction map; its nonempty branch is not implemented.
 if(type==1&&field<std::uint64_t>(c,216)!=field<std::uint64_t>(c,224))throw std::runtime_error("pixel correction map unsupported");
 double f=field<double>(c,24),cx=field<double>(c,32),cy=field<double>(c,40),b1=field<double>(c,48),b2=field<double>(c,56);
 double y=(p[1]-double(field<std::uint64_t>(c,16))*0.5-cy)/f;
 double x=(p[0]-double(field<std::uint64_t>(c,8))*0.5-cx-y*b2)/(f+b1);
 if(type==1&&field<unsigned char>(c,72)){
  std::array<double,12> d;for(unsigned i=0;i<12;++i)d[i]=field<double>(c,80+8*i);
  double initial_x=x,initial_y=y,previous_x=x,previous_y=y;
  for(unsigned iteration=0;iteration<10;++iteration){
   double r=y*y+x*x;
   double inverse=1.0/(d[0]*r+1.0+d[1]*(r*r)+d[2]*r*(r*r)+d[3]*(r*r)*(r*r));
   double tangential=d[6]*r+1.0+d[7]*(r*r),prism=d[10]*r+1.0+d[11]*(r*r);
   double dy=((y*2.0*y+r)*d[5]+d[4]*2.0*x*y)*tangential+d[9]*prism*r;
   double dx=d[5]*2.0*x*y;
   double old_x=x;y=(initial_y-dy)*inverse;
   x=(initial_x-(((old_x*2.0*old_x+r)*d[4]+dx)*tangential+d[8]*prism*r))*inverse;
   if(iteration>=4&&(y-previous_y)*(y-previous_y)+(x-previous_x)*(x-previous_x)<1.0e-10)break;
   previous_x=x;previous_y=y;
  }
 }
 return {x,y,1.0};
}
