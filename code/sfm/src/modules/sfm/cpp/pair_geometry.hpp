#pragma once
#include "projection_decompose.hpp"
#include "triangulate.hpp"
#include <span>
#include <vector>
namespace photo_initial {
using Projection=std::array<double,12>;
using Pixel=std::array<double,2>;
// 143A74590: P=K[R|-R*C], preserving the machine-code addition order.
inline Projection compose(const Camera&c){
 std::array<double,3>t;for(unsigned i=0;i<3;++i)t[i]=(c.center[0]*-c.rotation[3*i]+c.center[1]*-c.rotation[3*i+1])+c.center[2]*-c.rotation[3*i+2];
 Projection p;for(unsigned i=0;i<3;++i){for(unsigned j=0;j<3;++j)p[4*i+j]=(c.rotation[3+j]*c.intrinsic[3*i+1]+c.rotation[j]*c.intrinsic[3*i])+c.rotation[6+j]*c.intrinsic[3*i+2];p[4*i+3]=(t[1]*c.intrinsic[3*i+1]+t[0]*c.intrinsic[3*i])+t[2]*c.intrinsic[3*i+2];}return p;
}
// 143A73070: first camera defines the coordinate frame; intrinsics are retained.
inline void canonicalize(Projection&first,Projection&second){
 auto a=decompose(first),b=decompose(second);auto rotation=multiply(b.rotation,transpose(a.rotation));std::array<double,3>d,c;
 for(unsigned i=0;i<3;++i)d[i]=b.center[i]-a.center[i];
 for(unsigned i=0;i<3;++i)c[i]=(d[1]*a.rotation[3*i+1]+d[0]*a.rotation[3*i])+d[2]*a.rotation[3*i+2];
 first=compose({a.intrinsic,{1,0,0,0,1,0,0,0,1},{0,0,0}});second=compose({b.intrinsic,rotation,c});
}
// 143AD2880: nine 4x4 determinants, expanded into six products of 2x2 minors.
inline Matrix fundamental(const Projection&p,const Projection&q){
 Matrix f;
 for(unsigned j=0;j<3;++j)for(unsigned i=0;i<3;++i){auto a=p.data()+4*((i+1)%3),b=p.data()+4*((i+2)%3),c=q.data()+4*((j+1)%3),d=q.data()+4*((j+2)%3);
  f[3*j+i]=(a[0]*b[1]-b[0]*a[1])*(c[2]*d[3]-c[3]*d[2])
   -(b[2]*a[0]-a[2]*b[0])*(c[1]*d[3]-d[1]*c[3])
   +(b[3]*a[0]-a[3]*b[0])*(c[1]*d[2]-d[1]*c[2])
   +(c[0]*d[3]-d[0]*c[3])*(b[2]*a[1]-a[2]*b[1])
   -(b[3]*a[1]-a[3]*b[1])*(c[0]*d[2]-d[0]*c[2])
   +(b[3]*a[2]-a[3]*b[2])*(c[0]*d[1]-d[0]*c[1]);
 }return f;
}
// 143AD9A00/143A73810 recompute every mask entry, including previous outliers.
inline std::size_t filter_pair(const Matrix&f,std::span<const Pixel>a,std::span<const Pixel>b,std::span<unsigned char>mask,double ta,double tb){
 if(a.size()!=b.size()||a.size()!=mask.size())throw std::runtime_error("pair filter extent");ta*=ta;tb*=tb;std::size_t changed=0;
 for(unsigned i=0;i<a.size();++i){double x=a[i][0],y=a[i][1],u=b[i][0],v=b[i][1];double l0=f[1]*y+f[0]*x+f[2],l1=f[4]*y+f[3]*x+f[5];double r=u*l0+v*l1+f[7]*y+f[6]*x+f[8];double h0=v*f[3]+u*f[0]+f[6],h1=v*f[4]+u*f[1]+f[7];double s=x*h0+h1*y+f[5]*v+f[2]*u+f[8];bool keep=!((l0*l0+l1*l1)*tb<=r*r||(h0*h0+h1*h1)*ta<=s*s);changed+=bool(mask[i])!=keep;mask[i]=keep;}
 return changed;
}
inline std::array<double,4> triangulate_pixels(const Projection&p,const Projection&q,const Pixel&a,const Pixel&b){double r[]{a[0],a[1],1},s[]{b[0],b[1],1};return ms::five::triangulate(p.data(),q.data(),r,s);}
}
