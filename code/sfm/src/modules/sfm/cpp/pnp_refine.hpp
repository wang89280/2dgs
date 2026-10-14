#pragma once
#include "pair_refine.hpp"
namespace sfm {
// 143AEA410: partial-pivot elimination. Ties choose the last row; singular
// back-substitution components become zero, as in the native implementation.
inline std::array<double,6> pnp_linear(std::array<double,36>a,std::array<double,6>b){
 for(unsigned i=0;i<6;++i){unsigned pivot=i;for(unsigned j=i+1;j<6;++j)if(std::abs(a[6*j+i])>=std::abs(a[6*pivot+i]))pivot=j;if(pivot!=i){for(unsigned k=i;k<6;++k)std::swap(a[6*i+k],a[6*pivot+k]);std::swap(b[i],b[pivot]);}if(a[6*i+i]==0)continue;double inverse=1/a[6*i+i];for(unsigned j=i+1;j<6;++j){double factor=inverse*a[6*j+i];for(unsigned k=i+1;k<6;++k)a[6*j+k]+=(-factor)*a[6*i+k];b[j]-=factor*b[i];}}
 for(unsigned i=6;i--;){if(a[6*i+i]==0){b[i]=0;continue;}double sum=0;for(unsigned k=i+1;k<6;++k)sum+=a[6*i+k]*b[k];b[i]=(b[i]-sum)/a[6*i+i];}return b;
}
// 142467BD0 / 142468540 / 143B2ABD0: Rodrigues vector and camera center,
// additive LM damping, ten total trial steps, relative parameter-change stop.
inline Pose refine_pose(Pose p,const std::array<std::byte,256>&cal,std::span<const V>x,std::span<const Obs>obs,std::span<const unsigned char>mask,double tolerance=1e-6,unsigned iterations=10){
 auto w=photo_initial::rotation_vector(p.rotation);auto c=center(p);std::array<double,6>g{w[0],w[1],w[2],c[0],c[1],c[2]};
 struct Eval{std::vector<double>r;std::vector<std::array<double,6>>j;double cost{};};
 auto evaluate_g=[&](const std::array<double,6>&v,bool derivatives){Eval e;auto rot=photo_initial::rodrigues({v[0],v[1],v[2]});Pose q;q.rotation=rot.r;set(q.position.data(),96,V{v[3],v[4],v[5]});for(unsigned i=0;i<x.size();++i)if(mask[i]){auto o=obs[i];o.size=1;auto local=sub(x[i],center(q));auto xyz=mv(rot.r,local);auto value=evaluate(cal.data(),q,x[i],o,false);for(unsigned row=0;row<2;++row)e.r.push_back(value.residual[row]);if(derivatives){auto d=calibration::evaluate(cal.data(),xyz.data());std::array<double,6>ju{},jv{};for(unsigned k=0;k<6;++k){V a;if(k<3)a=mv(rot.j[k],local);else a={-rot.r[k-3],-rot.r[k],-rot.r[k+3]};ju[k]=(a[1]*d.point[1]+a[0]*d.point[0])+a[2]*d.point[2];jv[k]=(a[1]*d.point[4]+a[0]*d.point[3])+a[2]*d.point[5];}e.j.push_back(ju);e.j.push_back(jv);}}for(double v:e.r)e.cost+=v*v;return e;};
 auto e=evaluate_g(g,true);if(e.r.size()<6)return p;double lambda=.001;unsigned attempts=0;bool stop=false;
 while(attempts<iterations&&!stop){std::array<double,36>a{};std::array<double,6>b{};for(unsigned row=0;row<e.r.size();++row)for(unsigned i=0;i<6;++i){b[i]+=e.r[row]*e.j[row][i];for(unsigned j=0;j<6;++j)a[6*i+j]+=e.j[row][j]*e.j[row][i];}
  while(attempts<iterations){++attempts;auto matrix=a;for(unsigned i=0;i<6;++i)matrix[7*i]+=lambda;auto delta=pnp_linear(matrix,b);auto next=g;for(unsigned i=0;i<6;++i)next[i]+=delta[i];auto value=evaluate_g(next,false);if(value.cost<e.cost){double change=0,norm=0;for(unsigned i=0;i<6;++i){double d=next[i]-g[i];change+=d*d;norm+=g[i]*g[i];}stop=std::sqrt(change/norm)<=tolerance;g=next;e=evaluate_g(g,true);lambda/=10;break;}lambda*=10;}
 }
 p.rotation=photo_initial::rodrigues({g[0],g[1],g[2]}).r;set(p.position.data(),96,V{g[3],g[4],g[5]});return p;
}
}
