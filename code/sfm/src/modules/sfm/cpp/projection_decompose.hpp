#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace photo_initial {
using Matrix=std::array<double,9>;
inline Matrix multiply(const Matrix&a,const Matrix&b){Matrix c{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)c[3*i+j]=(a[3*i]*b[j]+a[3*i+1]*b[3+j])+a[3*i+2]*b[6+j];return c;}
inline Matrix transpose(const Matrix&a){return {a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]};}
inline double determinant(const Matrix&a){return (a[4]*a[8]-a[5]*a[7])*a[0]-(a[3]*a[8]-a[6]*a[5])*a[1]+(a[3]*a[7]-a[6]*a[4])*a[2];}
inline Matrix inverse(const Matrix&a){double d=determinant(a);if(d!=0)d=1.0/d;return {(a[8]*a[4]-a[5]*a[7])*d,(a[2]*a[7]-a[1]*a[8])*d,(a[1]*a[5]-a[2]*a[4])*d,(a[6]*a[5]-a[3]*a[8])*d,(a[8]*a[0]-a[6]*a[2])*d,(a[3]*a[2]-a[5]*a[0])*d,(a[3]*a[7]-a[6]*a[4])*d,(a[1]*a[6]-a[0]*a[7])*d,(a[0]*a[4]-a[3]*a[1])*d};}
struct Camera {Matrix intrinsic,rotation;std::array<double,3> center;};
// 143A128A0: three Givens rotations in the original evaluation order.
inline Camera decompose(const std::array<double,12>&p){
 Matrix m{p[0],p[1],p[2],p[4],p[5],p[6],p[8],p[9],p[10]},a=m;
 if(m[6]==0&&m[7]==0&&m[8]==0)throw std::runtime_error("affine projection branch not implemented");
 double n=std::sqrt(a[8]*a[8]+a[7]*a[7]);if(!(n>0))throw std::runtime_error("singular RQ first rotation");
 double y=a[7]/n,z=-a[8]/n;Matrix xrot{1,0,0,0,z,-y,0,y,z};a=multiply(a,xrot);a[7]=0;
 n=std::sqrt(a[6]*a[6]+a[8]*a[8]);if(!(n>0))throw std::runtime_error("singular RQ second rotation");
 double x=a[6]/n;z=a[8]/n;Matrix yrot{z,0,x,0,1,0,-x,0,z};a=multiply(a,yrot);a[6]=0;
 n=std::sqrt(a[4]*a[4]+a[3]*a[3]);if(!(n>0))throw std::runtime_error("singular RQ third rotation");
 double q=a[3]/n,t=-a[4]/n;Matrix zrot{t,-q,0,q,t,0,0,0,1};a=multiply(a,zrot);a[3]=0;
 Camera c{a,multiply(multiply(transpose(zrot),transpose(yrot)),transpose(xrot)),{}};
 auto inv=inverse(m);for(unsigned i=0;i<3;++i)c.center[i]=p[7]*-inv[3*i+1]+p[3]*-inv[3*i]+p[11]*-inv[3*i+2];
 if(c.intrinsic[0]<0){c.intrinsic[0]=-c.intrinsic[0];for(unsigned j=0;j<3;++j)c.rotation[j]=-c.rotation[j];}
 if(c.intrinsic[4]<0){c.intrinsic[4]=-c.intrinsic[4];c.intrinsic[1]=-c.intrinsic[1];for(unsigned j=3;j<6;++j)c.rotation[j]=-c.rotation[j];}
 if(c.intrinsic[8]<0){c.intrinsic[2]=-c.intrinsic[2];c.intrinsic[5]=-c.intrinsic[5];c.intrinsic[8]=-c.intrinsic[8];for(unsigned j=6;j<9;++j)c.rotation[j]=-c.rotation[j];}
 if(determinant(c.rotation)<0){for(auto&v:c.intrinsic)v=-v;for(auto&v:c.rotation)v=-v;}return c;
}
}
