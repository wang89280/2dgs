#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace calibration {
template<class T> T get(const std::byte* c,unsigned offset){T x;std::memcpy(&x,c+offset,sizeof x);return x;}
struct Derivatives {std::array<double,6> point{};std::array<std::array<double,2>,18> parameter{};};
// Current dataset's frame camera branch: 142347710 -> 142642560,
// 14234AD40 -> 142642B40. Native slot 5 is unused in this branch.
// Retains the native Jacobian's radial lower clamp (0.5), which is not
// the derivative of the separately capped forward projection in all domains.
inline Derivatives evaluate(const std::byte* c,const double* xyz){
 if(get<int>(c,0)!=1||get<std::uint64_t>(c,240)||get<std::uint64_t>(c,216)!=get<std::uint64_t>(c,224)||get<double>(c,64)!=0)
  throw std::runtime_error("unvalidated camera/correction model");
 if(std::to_integer<unsigned>(c[72])!=1)throw std::runtime_error("requires native distortion-enabled frame camera");
 double f=get<double>(c,24),b1=get<double>(c,48),b2=get<double>(c,56);
 std::array<double,13> k{};for(unsigned i=0;i<13;++i)k[i]=get<double>(c,80+i*8);
 double iz=xyz[2];if(iz!=0)iz=1.0/iz;
 double x=xyz[0]*iz,y=xyz[1]*iz,r=y*y+x*x,r2=r*r,r3=r2*r,r4=r2*r2;
 double t=r*k[6]+1.0+r2*k[7],s=r*k[10]+1.0+r2*k[11];
 double radial=std::fmax(0.5,r*k[0]+1.0+r2*k[1]+r3*k[2]+r4*k[3]);
 double xx=x*2.0*x+r,yy=y*2.0*y+r;
 double tx=k[5]*2.0*x*y+k[4]*xx,ty=k[4]*2.0*x*y+yy*k[5];
 Derivatives out;auto& p=out.parameter;
 p[6]={r*x,r*y};p[7]={r2*x,r2*y};p[8]={r3*x,r3*y};p[9]={r4*x,r4*y};
 p[10]={xx*t,(x*2.0*y)*t};p[11]={(x*2.0*y)*t,yy*t};
 p[12]={tx*r,ty*r};p[13]={tx*r2,ty*r2};p[14]={s*r,s*0.0};p[15]={s*0.0,s*r};
 p[16]={(k[8]*r)*r,(k[9]*r)*r};p[17]={(k[8]*r)*r2,(k[9]*r)*r2};
 double dr=k[1]*2.0*r+k[0]+k[2]*3.0*r2+k[3]*4.0*r3;
 double dt=k[7]*2.0*r+k[6],ds=k[11]*2.0*r+k[10];
 double vx=y*0.0+x,vy=x*0.0+y,dx=vx*2.0,dy=vy*2.0;
 double sx=k[8]*s,sy=k[9]*s;
 double j00=((dx+x*4.0)*k[4]+vy*(k[5]*2.0))*t+dx*dr*x+radial+dx*dt*tx+dx*sx+k[8]*(dx*ds)*r;
 double j10=((y*4.0*0.0+dx)*k[5]+vy*(k[4]*2.0))*t+dx*dr*y+radial*0.0+dx*dt*ty+dx*sy+k[9]*(dx*ds)*r;
 double j01=((x*4.0*0.0+dy)*k[4]+vx*(k[5]*2.0))*t+dy*dr*x+radial*0.0+dy*dt*tx+dy*sx+k[8]*(dy*ds)*r;
 double j11=((dy+y*4.0)*k[5]+vx*(k[4]*2.0))*t+dy*dr*y+radial+dy*dt*ty+dy*sy+(k[9]*(dy*ds))*r;
 double n00=iz*j00+j01*0.0,n10=iz*j10+j11*0.0;
 double n01=j01*iz+j00*0.0,n11=iz*j11+j10*0.0;
 double nz0=-x*iz,nz1=-y*iz;
 double n02=nz1*j01+j00*nz0,n12=nz1*j11+j10*nz0;
 double fx=f+b1;
 out.point={n00*fx+b2*n10,b2*n11+fx*n01,n12*b2+fx*n02,n00*0.0+f*n10,f*n11+0.0*n01,n12*f+0.0*n02};
 // Forward distortion is used for d(pixel)/d(f,b1,b2), including radius cap.
 double rc=x*x+y*y,cap_scale=1.0;if(rc>k[12]){cap_scale=k[12]/rc;rc=k[12];}
 double radial_delta=rc*rc*k[1]+rc*k[0]+rc*k[2]*(rc*rc)+rc*rc*k[3]*(rc*rc);
 double tc=rc*k[6]+1.0+rc*rc*k[7],sc=rc*k[10]+1.0+rc*rc*k[11];
 double xd=x+((x*3.0*x+y*y)*k[4]+k[5]*2.0*x*y)*tc*cap_scale+radial_delta*x+rc*k[8]*sc*cap_scale;
 double yd=y+((y*3.0*y+x*x)*k[5]+k[4]*2.0*x*y)*tc*cap_scale+radial_delta*y+rc*k[9]*sc*cap_scale;
 p[0]={xd,yd};p[1]={1,0};p[2]={0,1};p[3]={xd,0};p[4]={yd,0};
 for(unsigned i=6;i<18;++i){double a=p[i][0],b=p[i][1];p[i]={a*fx+b*b2,a*0.0+b*f};}
 return out;
}
}
