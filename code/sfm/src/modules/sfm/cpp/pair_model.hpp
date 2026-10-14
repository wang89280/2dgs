#pragma once
#include "pair_geometry.hpp"
namespace photo_initial {
struct RotationModel{Matrix r;std::array<Matrix,3>j;};
// 143A14780: Rodrigues rotation and its analytic derivatives.
inline RotationModel rodrigues(const std::array<double,3>&w){
 RotationModel out{};double theta=std::sqrt(w[1]*w[1]+w[0]*w[0]+w[2]*w[2]);
 constexpr std::array<Matrix,3> generators{{{0,0,0,0,0,-1,0,1,0},{0,0,1,0,0,0,-1,0,0},{0,-1,0,1,0,0,0,0,0}}};
 if(theta<2.220446049250313e-16){out.r={1,0,0,0,1,0,0,0,1};out.j=generators;return out;}
 double c=std::cos(theta),s=std::sin(theta),v=1-c,inv=1/theta;std::array<double,3>u{w[0]*inv,w[1]*inv,w[2]*inv};Matrix skew{0,-u[2],u[1],u[2],0,-u[0],-u[1],u[0],0};
 Matrix outer{u[0]*u[0],u[1]*u[0],u[2]*u[0],u[1]*u[0],u[1]*u[1],u[2]*u[1],u[2]*u[0],u[2]*u[1],u[2]*u[2]};
 for(unsigned r=0;r<3;++r)for(unsigned col=0;col<3;++col)out.r[3*r+col]=outer[3*r+col]*v+c*double(r==col)+s*skew[3*r+col];
 double a=inv*v,b=s-v*2.0*inv,d=c-inv*s,e=inv*s;
 for(unsigned k=0;k<3;++k)for(unsigned r=0;r<3;++r)for(unsigned col=0;col<3;++col){unsigned n=3*r+col;double du=(r==k?u[col]:0)+(col==k?u[r]:0);double aa=a*du,bb=b*u[k]*outer[n],cc=-s*u[k]*double(r==col),dd=d*u[k]*skew[n],ee=e*generators[k][n];out.j[k][n]=(n==1?(bb+cc)+aa:aa+(bb+cc))+dd+ee;}
 return out;
}
struct PairEvaluation{std::vector<std::array<double,4>>model;std::vector<std::array<double,36>>jacobian;};
// 143A75580 / 143A74EB0 use inverse-depth point coordinates (X/Z,Y/Z,W/Z).
// Preserve the captured rotation derivative convention; do not substitute a
// differently parameterized bundle-adjustment Jacobian.
inline PairEvaluation evaluate_pair(const std::array<double,6>&g,std::span<const std::array<double,3>>points,const std::array<Matrix,2>&k){
 auto rot=rodrigues({g[0],g[1],g[2]});auto p=compose({k[0],{1,0,0,0,1,0,0,0,1},{0,0,0}}),q=compose({k[1],rot.r,{g[3],g[4],g[5]}});for(auto&j:rot.j)j=multiply(k[1],j);
 PairEvaluation out;out.model.resize(points.size());out.jacobian.resize(points.size());
 for(unsigned i=0;i<points.size();++i){auto&x=points[i];double px=x[0]*p[0]+x[1]*p[1]+p[2]+x[2]*p[3],py=x[0]*p[4]+x[1]*p[5]+p[6]+x[2]*p[7],ip=1/(x[0]*p[8]+x[1]*p[9]+p[10]+x[2]*p[11]);double qx=x[0]*q[0]+x[1]*q[1]+q[2]+x[2]*q[3],qy=x[0]*q[4]+x[1]*q[5]+q[6]+x[2]*q[7],iq=1/(x[0]*q[8]+x[1]*q[9]+q[10]+x[2]*q[11]);out.model[i]={px*ip,ip*py,qx*iq,iq*qy};auto&j=out.jacobian[i];
  for(unsigned n=0;n<3;++n){auto&d=rot.j[n];double dx=x[0]*d[0]+x[1]*d[1]+d[2],dy=x[0]*d[3]+x[1]*d[4]+d[5],dz=x[0]*d[6]+x[1]*d[7]+d[8];j[4*n+2]=dx*iq-dz*qx*iq*iq;j[4*n+3]=dy*iq-dz*qy*iq*iq;}
  for(unsigned n=0;n<3;++n){double dx=-q[n]*x[2],dy=-q[4+n]*x[2],dz=-q[8+n]*x[2];j[4*(n+3)+2]=dx*iq-dz*qx*iq*iq;j[4*(n+3)+3]=dy*iq-dz*qy*iq*iq;}
  for(unsigned n=0;n<3;++n){unsigned c=n<2?n:3;j[24+4*n]=ip*p[c]-p[8+c]*px*ip*ip;j[25+4*n]=ip*p[4+c]-p[8+c]*py*ip*ip;j[26+4*n]=iq*q[c]-q[8+c]*qx*iq*iq;j[27+4*n]=iq*q[4+c]-q[8+c]*qy*iq*iq;}
 }return out;
}
}
