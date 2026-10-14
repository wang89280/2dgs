#pragma once
#include "pair_geometry.hpp"
namespace photo_initial {
using Point3=std::array<double,3>;
// 143A6FE70 accumulates A^T A across every view before its 4x4 SVD.
inline std::array<double,4> triangulate_views(std::span<const Projection>p,std::span<const Point3>r){
 if(p.size()!=r.size()||p.size()<2)throw std::runtime_error("triangulation extent");
 std::array<double,16>a{};for(unsigned v=0;v<p.size();++v)for(unsigned j=0;j<2;++j){std::array<double,4>row;for(unsigned k=0;k<4;++k)row[k]=r[v][j]*p[v][8+k]-r[v][2]*p[v][4*j+k];for(unsigned k=0;k<4;++k)for(unsigned l=0;l<4;++l)a[4*k+l]+=row[k]*row[l];}
 auto s=ms::five::square_svd<4>(a);return {s.vt[12],s.vt[13],s.vt[14],s.vt[15]};
}
inline int quadratic(double a,double b,double*out){double h=a*.5,d=h*h-b;if(d<0)return 0;if(d==0){out[0]=-h;return 1;}double s=std::sqrt(d);out[0]=s-h;out[1]=-h-s;return 2;}
// 143B34300, including the native pow exponent rather than cbrt.
inline int cubic(const double*p,double*out){double a=p[3],b=p[2],c=p[1],d=p[0];out[0]=out[1]=out[2]=0;
 if(a==0){if(b==0){if(c==0)return d==0?-1:0;out[0]=-d/c;return 1;}double disc=c*c-b*4*d;if(disc<0)return 0;double s=std::sqrt(disc),v=((c<0?-s:s)-c)*.5;out[0]=v/b;out[1]=d/v;return 1+(s>0);}
 double inv=1/a; b*=inv;c*=inv;double q=(b*b-c*3)*0.1111111111111111,r=(b*2*b*b-b*9*c+d*inv*27)*0.01851851851851852,q3=q*q*q,disc=q3-r*r;
 if(disc<0){double v=std::pow(std::sqrt(-disc)+std::abs(r),0.333333333333);if(r>0)v=-v;out[0]=q/v+v-b*0.3333333333333333;return 1;}
 double m=std::sqrt(q)*-2,shift=b*0.3333333333333333,t=std::acos(r/std::sqrt(q3))*0.3333333333333333;out[0]=std::cos(t)*m-shift;out[1]=std::cos(t+2.094395102393195)*m-shift;out[2]=std::cos(t+4.188790204786391)*m-shift;return 3;
}
// 143B34710: coefficients are in ascending order.
inline int quartic(const std::array<double,5>&c,double*out){if(c[4]==0)return cubic(c.data(),out);double b=c[3]/c[4]*.25,c2=c[2]/c[4],c1=c[1]/c[4];double p=c2-b*6*b,q=(b*8*b-c2*2)*b+c1,r=((c2-b*3*b)*b-c1)*b+c[0]/c[4];int n=0;
 if(q==0){double d=p*p-r*4;if(d<0)return 0;double s=std::sqrt(d),v=(s-p)*.5;if(v<0)return 0;v=std::sqrt(v);out[n++]=v-b;out[n++]=-v-b;if(s>0){v=(-p-s)*.5;if(v>=0){v=std::sqrt(v);out[n++]=v-b;out[n++]=-v-b;}}return n;}
 if(r==0){double coeff[]{q,p,0,1};n=cubic(coeff,out);if(n<0)n=0;for(int i=0;i<n;++i)out[i]-=b;out[n++]=-b;return n;}
 double coeff[]{r*.5*p-q*.125*q,-r,p*-.5,1};if(cubic(coeff,out)==-1)out[0]=0;double z=out[0],d=z*z-r,e=z*2-p;if(d<0||e<0)return 0;double u=std::sqrt(d),v=std::sqrt(e);if(q<0)v=-v;n=quadratic(v,z-u,out);n+=quadratic(-v,z+u,out+n);for(int i=0;i<n;++i)out[i]-=b;return n;
}
inline Camera rigid_three(const std::array<Point3,3>&x,const std::array<Point3,3>&y){Point3 mx{},my{};for(unsigned k=0;k<3;++k){for(unsigned i=0;i<3;++i){mx[k]+=x[i][k];my[k]+=y[i][k];}mx[k]/=3;my[k]/=3;}
 Matrix h{};for(unsigned i=0;i<3;++i)for(unsigned k=0;k<3;++k)for(unsigned l=0;l<3;++l)h[k*3+l]+=(x[i][k]-mx[k])*(y[i][l]-my[l]);auto svd=ms::five::square_svd<3>(h);auto ut=svd.ut;auto vt=svd.vt;auto det=[](const Matrix&a){return (a[4]*a[8]-a[5]*a[7])*a[0]-(a[3]*a[8]-a[5]*a[6])*a[1]+(a[3]*a[7]-a[4]*a[6])*a[2];};if(det(ut)*det(vt)<=0)for(unsigned i=0;i<3;++i)ut[6+i]=-ut[6+i];auto r=multiply(transpose(vt),ut);Point3 center;for(unsigned k=0;k<3;++k)center[k]=mx[k]-((my[1]*r[3+k]+my[0]*r[k])+my[2]*r[6+k]);return {{1,0,0,0,1,0,0,0,1},r,center};
}
// 142465260: three-point pose candidates and the actual degeneracy gates.
inline std::vector<Projection> p3p(const std::array<Point3,3>&x,const std::array<Point3,3>&r,std::array<double,5>*polynomial=nullptr){auto length=[](Point3 a,Point3 b){double dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2];return dx*dx+dy*dy+dz*dz;};double ab=length(x[0],x[1]),bc=length(x[1],x[2]),ac=length(x[0],x[2]);if(ab<1e-8||bc<1e-8||ac<1e-8)return {};auto dot=[](Point3 a,Point3 b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};double na=1/std::sqrt(dot(r[0],r[0])),nb=1/std::sqrt(dot(r[1],r[1])),nc=1/std::sqrt(dot(r[2],r[2]));double ca=dot(r[1],r[0])*(nb*na),cb=dot(r[2],r[1])*(nc*nb),cc=dot(r[2],r[0])*(nc*na);if(ca>0.9999999899999999||cb>0.9999999899999999||cc>0.9999999899999999)return {};
 double a=bc/ab,b=bc/ac,t=a*b-b,u=a*b+b-a;std::array<double,5>p;
 p[4]=(t-a)*(t-a)-b*4*a*(cb*cb);
 p[3]=((a*b+a-b)*cc+a*2*ca*cb)*(b*4*cb)+(t-a)*4*a*(1-b)*ca;
 p[2]=((b-a)*(cb*cb)+(1-a)*b*(cc*cc)-(b+1)*(a*2)*ca*cc*cb)*(b*4)+(u*2*(t-a)+(1-b)*(a*2)*ca*((1-b)*(a*2)*ca));
 p[1]=(b*2*a*ca*(cc*cc)+(t+a)*cc*cb)*(b*4)+u*4*a*(1-b)*ca;
 p[0]=u*u-b*b*4*a*(cc*cc);if(polynomial)*polynomial=p;double roots[4];int n=quartic(p,roots);std::vector<Projection>out;
 for(int i=0;i<n;++i){double z=roots[i];if(!(z>0))continue;double z2=z*z,d1=std::sqrt(ab/(z2-z*2*ca+1)),d2=d1*z,v=z*2*a*ca+z2*(1-a)-a;if(z2-b==v*(1-b))continue;double d3=(-z*cb*2*(z2-b)-(b*cc-z*cb)*2*v)*d1/(v*(1-b)-(z2-b));std::array<Point3,3>y;for(unsigned k=0;k<3;++k){y[0][k]=r[0][k]*d1*na;y[1][k]=r[1][k]*d2*nb;y[2][k]=r[2][k]*d3*nc;}out.push_back(compose(rigid_three(x,y)));}return out;
}
}
