#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace ms::five {
using P1=std::array<double,4>;using P2=std::array<double,10>;using P3=std::array<double,20>;using Poly=std::array<double,11>;
// 143A1A000: only the trailing four left singular vectors of the transposed
// 9x5 epipolar design are needed. The subsequent QR rotations affect rows 0..4.
inline void house_rows(int rows,int cols,const double* h,int step,double* target,double factor){for(int i=1;i<rows;++i){auto row=target+i*step;double s=0;int j=0;for(;j+3<cols;j+=4)s=s+((h[j]*row[j]+h[j+1]*row[j+1])+h[j+2]*row[j+2]+h[j+3]*row[j+3]);for(;j<cols;++j)s=s+h[j]*row[j];s=s*factor;row[-1]=s*h[-1];for(int k=0;k<cols;++k)row[k]=s*h[k]+row[k];}}
inline std::array<double,81> nullspace(const double* first,const double* second,std::array<double,81>* reflectors=nullptr,std::vector<std::array<double,81>>*steps=nullptr){
 std::array<double,45>a{};std::array<double,81>u{};double v[25]{},initial=0;
 for(int i=0;i<5;++i)for(int r=0;r<3;++r)for(int c=0;c<3;++c)a[(r*3+c)*5+i]=second[i*3+r]*first[i*3+c];
 for(int i=0;i<5;++i){int nr=9-i,nc=5-i;auto p=u.data()+10*i;double scale=0,s=0,factor=0;
  for(int j=0;j<nr;++j){p[j]=a[(i+j)*5+i];scale=scale+std::abs(p[j]);}
  if(scale!=0){double inv=1/scale;for(int j=0;j<nr;++j){p[j]=inv*p[j];s=s+p[j]*p[j];}double g=std::sqrt(s);if(p[0]>=0)g=-g;double f=p[0];p[0]=f-g;factor=1/(g*f-s);double t[5]{};
   for(int r=0;r<nr;++r)for(int c=1;c<nc;++c)t[c]=p[r]*a[(i+r)*5+i+c]+t[c];
   for(int c=1;c<nc;++c)t[c]=factor*t[c];
   for(int r=0;r<nr;++r)for(int c=1;c<nc;++c)a[(i+r)*5+i+c]=p[r]*t[c]+a[(i+r)*5+i+c];}
  if(i==0)initial=factor;else p[-1]=factor;
  if(nc==1)break;auto q=v+6*(i+1);scale=0;s=0;factor=0;
  for(int j=0;j<nc-1;++j){q[j]=a[i*5+i+1+j];scale=scale+std::abs(q[j]);}
  if(scale!=0){double inv=1/scale;for(int j=0;j<nc-1;++j){q[j]=inv*q[j];s=s+q[j]*q[j];}double g=std::sqrt(s);if(q[0]>=0)g=-g;double f=q[0];q[-1]=0;q[0]=f-g;factor=1/(g*f-s);house_rows(nr,nc-1,q,5,a.data()+i*5+i+1,factor);}q[-1]=factor;
 }
 if(reflectors)*reflectors=u;
 for(int i=8;i>=0;--i){int n=9-i;auto p=u.data()+10*i;double factor=i>=5?0:(i?p[-1]:initial);if(factor==0){for(int j=1;j<n;++j){p[j]=0;p[j*9]=0;}p[0]=1;}else{house_rows(n,n-1,p+1,9,p+1,factor);if(steps)steps->push_back(u);factor=factor*p[0];for(int j=0;j<n;++j)p[j]=factor*p[j];p[0]=p[0]+1;}}
 return u;
}
// Coefficient layouts and summation order recovered from 143B8C270/143B8C3A0.
inline P2 mul(const P1&a,const P1&b){return {b[0]*a[0],a[1]*b[0]+b[1]*a[0],a[2]*b[0]+b[2]*a[0],a[3]*b[0]+b[3]*a[0],a[1]*b[1],a[2]*b[2],a[3]*b[3],b[2]*a[1]+a[2]*b[1],b[3]*a[1]+a[3]*b[1],b[3]*a[2]+a[3]*b[2]};}
inline P3 mul(const P2&a,const P1&b){P3 r{};
 r[16]=b[0]*a[0];r[10]=a[1]*b[0]+b[1]*a[0];r[13]=a[2]*b[0]+b[2]*a[0];r[17]=a[3]*b[0]+b[3]*a[0];
 r[9]=b[2]*a[1]+a[2]*b[1]+a[7]*b[0];r[11]=b[3]*a[1]+a[3]*b[1]+a[8]*b[0];r[14]=b[3]*a[2]+a[3]*b[2]+a[9]*b[0];
 r[5]=a[4]*b[0]+a[1]*b[1];r[7]=a[5]*b[0]+a[2]*b[2];r[18]=a[3]*b[3]+a[6]*b[0];
 r[0]=a[4]*b[1];r[2]=a[4]*b[2]+a[7]*b[1];r[4]=a[4]*b[3]+a[8]*b[1];r[3]=a[5]*b[1]+a[7]*b[2];r[1]=a[5]*b[2];r[6]=a[5]*b[3]+a[9]*b[2];r[12]=a[8]*b[3]+a[6]*b[1];r[19]=a[6]*b[3];r[8]=a[7]*b[3]+a[8]*b[2]+a[9]*b[1];r[15]=a[6]*b[2]+a[9]*b[3];return r;}
template<std::size_t N>inline std::array<double,N> add(const std::array<double,N>&a,const std::array<double,N>&b){auto r=a;for(int i=0;i<N;++i)r[i]=a[i]+b[i];return r;}
template<std::size_t N>inline std::array<double,N> sub(const std::array<double,N>&a,const std::array<double,N>&b){auto r=a;for(int i=0;i<N;++i)r[i]=a[i]-b[i];return r;}
inline Poly mul(const Poly&a,const Poly&b){Poly r{};for(int i=0;i<11;++i)for(int j=0;j<11-i;++j)r[i+j]=a[i]*b[j]+r[i+j];return r;}
struct Algebra{std::array<double,81> basis;std::array<P3,10> equations,reduced;Poly nx,ny,den,polynomial;};
inline Algebra algebra_basis(const std::array<double,81>& basis){Algebra out{};out.basis=basis;P1 e[9];for(int i=0;i<9;++i)e[i]={out.basis[72+i],out.basis[45+i],out.basis[54+i],out.basis[63+i]};
 P2 s[9];for(int r=0;r<3;++r)for(int c=r;c<3;++c){s[r*3+c]=add(add(mul(e[r*3+1],e[c*3+1]),mul(e[r*3],e[c*3])),mul(e[r*3+2],e[c*3+2]));s[c*3+r]=s[r*3+c];}
 auto trace=add(s[8],add(s[4],s[0]));for(auto&v:trace)v=v*.5;for(int i=0;i<3;++i)s[i*3+i]=sub(s[i*3+i],trace);
 for(int r=0;r<3;++r)for(int c=0;c<3;++c)out.equations[r*3+c]=add(add(mul(s[r*3],e[c]),mul(s[r*3+1],e[3+c])),mul(s[r*3+2],e[6+c]));
 out.equations[9]=add(add(mul(sub(mul(e[1],e[5]),mul(e[2],e[4])),e[6]),mul(sub(mul(e[2],e[3]),mul(e[0],e[5])),e[7])),mul(sub(mul(e[0],e[4]),mul(e[1],e[3])),e[8]));
 out.reduced=out.equations;auto&m=out.reduced;
 for(int i=0;i<10;++i){double inv=1/m[i][i];for(auto&x:m[i])x=inv*x;for(int j=i+1;j<10;++j){double f=-m[j][i];if(f!=0)for(int k=0;k<20;++k)m[j][k]=f*m[i][k]+m[j][k];}}
 for(int i=9;i>=4;--i)for(int j=0;j<i;++j){double f=-m[j][i];if(f!=0)for(int k=0;k<20;++k)m[j][k]=f*m[i][k]+m[j][k];}
 Poly p[3][3]{};for(int i=0;i<3;++i)for(int j=0;j<3;++j){int start=10+j*3,n=j==2?4:3;p[i][j][0]=m[4+i*2][start]-0.;for(int k=1;k<n;++k)p[i][j][k]=m[4+i*2][start+k]-m[5+i*2][start+k-1];p[i][j][n]=0.-m[5+i*2][start+n-1];}
 out.nx=sub(mul(p[0][1],p[1][2]),mul(p[0][2],p[1][1]));out.ny=sub(mul(p[0][2],p[1][0]),mul(p[0][0],p[1][2]));out.den=sub(mul(p[0][0],p[1][1]),mul(p[0][1],p[1][0]));out.polynomial=add(add(mul(out.nx,p[2][0]),mul(out.ny,p[2][1])),mul(out.den,p[2][2]));return out;
}
inline Algebra algebra(const double*a,const double*b){return algebra_basis(nullspace(a,b));}
}
