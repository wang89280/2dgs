#pragma once
#include "pair_model.hpp"
#include <functional>
namespace photo_initial {
struct PairSingular:std::runtime_error{using std::runtime_error::runtime_error;};
inline double dot(const double*a,const double*b,unsigned n,unsigned sa=1,unsigned sb=1){double s=0;for(unsigned i=0;i<n;++i)s+=a[i*sa]*b[i*sb];return s;}
inline double cholesky_dot(const double*a,const double*b,unsigned n,bool peel){double even=0,odd=0,head=0;unsigned i=0;if(peel&&n){head=a[0]*b[0];i=1;}for(;i+1<n;i+=2){even+=a[i]*b[i];odd+=a[i+1]*b[i+1];}double s=even+head+odd;if(i<n)s+=a[i]*b[i];return s;}
// 143AE9CB0 / 143AEA090, lower Cholesky and two triangular substitutions.
// Native workspace Schur/RHS addresses are malloc_base + 488*N+336 / 776*N+624.
// Odd N peels one scalar in SSE2 DDOT. Reproduce this logical alignment without
// depending on the address chosen for this C++ stack or heap.
inline std::array<double,6> pair_linear(std::array<double,36>a,std::array<double,6>b,bool peel=false){
 for(unsigned i=0;i<6;++i){double d=a[6*i+i]-cholesky_dot(a.data()+6*i,a.data()+6*i,i,peel);if(!(d>0))throw PairSingular("pair Schur not positive definite");a[6*i+i]=std::sqrt(d);double inv=1/a[6*i+i];for(unsigned j=i+1;j<6;++j)a[6*j+i]=(a[6*j+i]-cholesky_dot(a.data()+6*j,a.data()+6*i,i,peel))*inv;}
 for(unsigned i=0;i<6;++i)b[i]=(b[i]-cholesky_dot(a.data()+6*i,b.data(),i,peel))/a[6*i+i];for(unsigned i=6;i--;)b[i]=(b[i]-dot(a.data()+6*(i+1)+i,b.data()+i+1,5-i,6,1))/a[6*i+i];return b;
}
struct PairSolution{std::array<double,6> global;std::vector<std::array<double,3>>points;unsigned attempts{},accepted{};double initial_cost{},final_cost{},change{};bool converged{};};
using LinearTrace=std::function<void(unsigned,const std::array<double,36>&,const std::array<double,6>&,const std::array<double,6>&)>;
inline PairSolution optimize_pair(std::array<double,6>global,std::vector<std::array<double,3>>points,const std::array<Matrix,2>&k,std::span<const std::array<double,4>>observations,double tolerance,unsigned iterations,LinearTrace trace={}){
 if(points.empty()||points.size()!=observations.size()||!iterations)throw std::runtime_error("pair optimize extent");
 PairSolution result;auto eval=evaluate_pair(global,points,k);auto residual=[&](const PairEvaluation&e){std::vector<std::array<double,4>>r(points.size());for(unsigned i=0;i<r.size();++i)for(unsigned j=0;j<4;++j)r[i][j]=observations[i][j]-e.model[i][j];return r;};auto cost=[](const auto&r){double s=0;for(auto&v:r)for(double x:v)s+=x*x;return s;};auto r=residual(eval);double current=cost(r),lambda=.001;result.initial_cost=current;
 while(result.attempts<iterations){
  std::array<double,36>u{};std::array<double,6>ga{};std::vector<Matrix>v(points.size());std::vector<std::array<double,18>>w(points.size());std::vector<std::array<double,3>>gb(points.size());
  for(unsigned i=0;i<points.size();++i){auto&j=eval.jacobian[i];for(unsigned a=0;a<6;++a){ga[a]+=dot(j.data()+4*a,r[i].data(),4);for(unsigned b=a;b<6;++b)u[6*a+b]+=dot(j.data()+4*a,j.data()+4*b,4);for(unsigned b=0;b<3;++b)w[i][3*a+b]=dot(j.data()+4*a,j.data()+24+4*b,4);}for(unsigned a=0;a<3;++a){gb[i][a]=dot(j.data()+24+4*a,r[i].data(),4);for(unsigned b=0;b<3;++b)v[i][3*a+b]=dot(j.data()+24+4*a,j.data()+24+4*b,4);}}
  bool accepted=false;
  while(result.attempts<iterations&&!accepted){++result.attempts;auto schur=u;auto rhs=ga;std::vector<Matrix>vi=v;std::vector<std::array<double,18>>wv(points.size());
   try{for(unsigned a=0;a<6;++a)schur[6*a+a]*=1+lambda;
    for(unsigned i=0;i<points.size();++i){for(unsigned a=0;a<3;++a)vi[i][3*a+a]*=1+lambda;if(determinant(vi[i])==0)throw PairSingular("singular pair point");vi[i]=inverse(vi[i]);for(unsigned a=0;a<6;++a)for(unsigned b=0;b<3;++b)wv[i][3*a+b]=dot(w[i].data()+3*a,vi[i].data()+b,3,1,3);for(unsigned a=0;a<6;++a){rhs[a]-=dot(wv[i].data()+3*a,gb[i].data(),3);for(unsigned b=a;b<6;++b)schur[6*a+b]-=dot(wv[i].data()+3*a,w[i].data()+3*b,3);}}
    for(unsigned a=0;a<6;++a)for(unsigned b=0;b<a;++b)schur[6*a+b]=schur[6*b+a];auto delta=pair_linear(schur,rhs,points.size()%2!=0);if(trace)trace(result.attempts-1,schur,rhs,delta);auto trial_g=global;auto trial_p=points;for(unsigned a=0;a<6;++a)trial_g[a]+=delta[a];for(unsigned i=0;i<points.size();++i){auto b=gb[i];for(unsigned a=0;a<3;++a)b[a]-=dot(w[i].data()+a,delta.data(),6,3,1);for(unsigned a=0;a<3;++a)trial_p[i][a]+=dot(vi[i].data()+3*a,b.data(),3);}
    auto trial_eval=evaluate_pair(trial_g,trial_p,k);auto trial_r=residual(trial_eval);double next=cost(trial_r);if(next<current){double change=0,norm=0;for(unsigned a=0;a<6;++a){double d=trial_g[a]-global[a];change+=d*d;norm+=global[a]*global[a];}for(unsigned i=0;i<points.size();++i)for(unsigned a=0;a<3;++a){double d=trial_p[i][a]-points[i][a];change+=d*d;norm+=points[i][a]*points[i][a];}result.change=std::sqrt(change/norm);global=trial_g;points=std::move(trial_p);eval=std::move(trial_eval);r=std::move(trial_r);current=next;lambda/=10;accepted=true;++result.accepted;if(result.change<=tolerance){result.converged=true;break;}}else lambda*=10;
   }catch(const PairSingular&){lambda*=10;}
  }
  if(result.converged)break;
 }
 result.global=global;result.points=std::move(points);result.final_cost=current;return result;
}
}
