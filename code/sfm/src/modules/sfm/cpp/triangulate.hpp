#pragma once
#include "five_point.hpp"
namespace ms::five {
inline double camera_hypot(double a,double b){a=std::abs(a);b=std::abs(b);if(a>b){double t=b/a;return std::sqrt(t*t+1)*a;}if(b==0)return a;double t=a/b;return std::sqrt(t*t+1)*b;}
template<int N>struct SquareSvd {std::array<double,N*N> ut,vt;std::array<double,N>w;};
template<int N>SquareSvd<N> square_svd(std::array<double,N*N>a){using Matrix=std::array<double,N*N>;Matrix ut{},vt{};
    double w[N]{},e[N]{},u0=0,v0=0;
    for(int i=0;i<N;++i){int n=N-i;auto u=ut.data()+(N+1)*i;double scale=0,s=0,factor=0;
        for(int j=0;j<n;++j){u[j]=a[(i+j)*N+i];scale=scale+std::abs(u[j]);}
        if(scale!=0){double inv=1/scale;for(int j=0;j<n;++j){u[j]=inv*u[j];s=s+u[j]*u[j];}double g=std::sqrt(s);if(u[0]>=0)g=-g;double f=u[0];u[0]=f-g;factor=1/(g*f-s);
            double t[N]{};for(int r=0;r<n;++r)for(int c=1;c<n;++c)t[c]=u[r]*a[(i+r)*N+i+c]+t[c];
            for(int c=1;c<n;++c)t[c]=factor*t[c];
            for(int r=0;r<n;++r)for(int c=1;c<n;++c)a[(i+r)*N+i+c]=u[r]*t[c]+a[(i+r)*N+i+c];
            w[i]=g*scale;}
        if(i==0)u0=factor;else u[-1]=factor;
        if(n==1)break;
        auto v=vt.data()+(N+1)*(i+1);scale=0;s=0;factor=0;
        for(int j=0;j<n-1;++j){v[j]=a[i*N+i+1+j];scale=scale+std::abs(v[j]);}
        if(scale!=0){double inv=1/scale;for(int j=0;j<n-1;++j){v[j]=inv*v[j];s=s+v[j]*v[j];}double g=std::sqrt(s);if(v[0]>=0)g=-g;double f=v[0];v[-1]=0;v[0]=f-g;factor=1/(g*f-s);
            house_rows(n,n-1,v,N,a.data()+i*N+i+1,factor);e[i+1]=g*scale;}
        v[-1]=factor;
    }
    auto form=[](Matrix& m,double first){for(int i=N-1;i>=0;--i){int n=N-i;auto p=m.data()+(N+1)*i;double factor=i?p[-1]:first;if(factor==0){for(int j=1;j<n;++j){p[j]=0;p[j*N]=0;}p[0]=1;}else{house_rows(n,n-1,p+1,N,p+1,factor);factor=factor*p[0];for(int j=0;j<n;++j)p[j]=factor*p[j];p[0]=p[0]+1;}}};
    form(ut,u0);form(vt,v0);
    double tolerance=0;for(int i=0;i<N;++i)tolerance=std::max(std::abs(w[i])+std::abs(e[i]),tolerance);tolerance=tolerance*2.220446049250313e-16;
    auto rotate=[](Matrix& m,int previous,int current,double c,double s){for(int j=0;j<N;++j){double a=m[previous*N+j],b=m[current*N+j];double next=b*c-a*s;m[previous*N+j]=b*s+a*c;m[current*N+j]=next;}};
    for(int k=N-1;k>=0;--k){int iteration=0;for(;;){int l=k;bool cancel=false;for(;l>=0;--l){if(!(tolerance<std::abs(e[l])))break;if(l==0||!(tolerance<std::abs(w[l-1]))){cancel=true;break;}}
            if(cancel){double c=0,s=1;for(int i=l;i<=k;++i){double f=e[i]*s;e[i]=e[i]*c;if(std::abs(f)+tolerance==tolerance)break;double g=w[i],h=camera_hypot(f,g);w[i]=h;s=-f/h;c=g/h;if(l==0)throw std::runtime_error("Camera SVD cancellation index");rotate(ut,l-1,i,c,s);}}
            double z=w[k];if(l==k){if(z<0){w[k]=-z;for(int j=0;j<N;++j)vt[k*N+j]=-vt[k*N+j];}break;}if(iteration++==30)throw std::runtime_error("Camera SVD did not converge");
            double y=w[k-1],h=e[k],x=w[l],f=((e[k-1]-z)/y*((e[k-1]+z)/h)+y/h-h/y)*.5;
            double g=camera_hypot(f,1);if(f<0)g=-g;
            f=(y/(g+f)-h)*(h/x)+(x-z/x*z);double c=1,s=1;
            for(int i=l+1;i<=k;++i){g=e[i];y=w[i];h=s*g;g=g*c;z=camera_hypot(f,h);e[i-1]=z;c=f/z;s=h/z;f=x*c+g*s;h=y*s;y=y*c;g=g*c-x*s;rotate(vt,i-1,i,c,s);z=camera_hypot(f,h);w[i-1]=z;if(z!=0){c=f/z;s=h/z;}f=y*s+g*c;x=y*c-g*s;rotate(ut,i-1,i,c,s);}
            e[l]=0;e[k]=f;w[k]=x;
        }}
    for(int i=0;i<N;++i){int best=i;for(int j=i+1;j<N;++j)if(w[best]<w[j])best=j;if(best!=i){std::swap(w[i],w[best]);for(int j=0;j<N;++j){std::swap(ut[i*N+j],ut[best*N+j]);std::swap(vt[i*N+j],vt[best*N+j]);}}}

SquareSvd<N> result{ut,vt};std::copy(w,w+N,result.w.begin());return result;}
inline std::array<double,4> triangulate(const double*p1,const double*p2,const double*r1,const double*r2){std::array<double,16>a;for(int k=0;k<4;++k){a[k]=r1[0]*p1[8+k]-r1[2]*p1[k];a[4+k]=r1[1]*p1[8+k]-r1[2]*p1[4+k];a[8+k]=r2[0]*p2[8+k]-r2[2]*p2[k];a[12+k]=r2[1]*p2[8+k]-r2[2]*p2[4+k];}auto svd=square_svd<4>(a);return {svd.vt[12],svd.vt[13],svd.vt[14],svd.vt[15]};}
}
