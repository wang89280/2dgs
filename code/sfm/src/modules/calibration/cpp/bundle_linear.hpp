#pragma once
#include <vector>
#include <span>
#include <cmath>
#include <stdexcept>
struct System {std::size_t n{};std::vector<double> a,b;std::vector<unsigned char> pattern;};
#ifdef MS_USE_CHOLMOD
#include "bundle_cholmod.hpp"
#else
inline std::vector<double> solve(const System& s){
 auto n=s.n;std::vector<double> scale(n),l(n*n);for(std::size_t i=0;i<n;++i){if(!(s.a[i*n+i]>0))throw std::runtime_error("nonpositive diagonal");scale[i]=std::sqrt(s.a[i*n+i]);}
 for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j){double sum=s.a[i*n+j]/scale[i]/scale[j];for(std::size_t k=0;k<j;++k)sum-=l[i*n+k]*l[j*n+k];if(i==j){if(!(sum>0))throw std::runtime_error("matrix not positive definite");l[i*n+j]=std::sqrt(sum);}else l[i*n+j]=sum/l[j*n+j];}
 auto triangular=[&](std::span<const double> rhs){std::vector<double> x(n);for(std::size_t i=0;i<n;++i){double v=rhs[i]/scale[i];for(std::size_t j=0;j<i;++j)v-=l[i*n+j]*x[j];x[i]=v/l[i*n+i];}for(std::size_t i=n;i--;){double v=x[i];for(std::size_t j=i+1;j<n;++j)v-=l[j*n+i]*x[j];x[i]=v/l[i*n+i];}for(std::size_t i=0;i<n;++i)x[i]/=scale[i];return x;};
 auto x=triangular(s.b);for(unsigned round=0;round<3;++round){std::vector<double> residual(n);for(std::size_t i=0;i<n;++i){double sum=0,correction=0;for(std::size_t j=0;j<n;++j){double y=s.a[i*n+j]*x[j]-correction,t=sum+y;correction=(t-sum)-y;sum=t;}residual[i]=s.b[i]-sum;}auto delta=triangular(residual);for(std::size_t i=0;i<n;++i)x[i]+=delta[i];}return x;
}
#endif
