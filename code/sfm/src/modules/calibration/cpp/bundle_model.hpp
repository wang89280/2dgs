#pragma once
#include "evidence.hpp"
#include "calibration_jacobian.hpp"
#include <bit>
namespace bundle {
using V=std::array<double,3>;using M=std::array<double,9>;
using calibration::get;
template<class T> void set(std::byte* p,unsigned offset,T x){std::memcpy(p+offset,&x,sizeof x);}
inline V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline double norm(V a){return std::sqrt(a[1]*a[1]+a[0]*a[0]+a[2]*a[2]);}
inline V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline V normalized(V a){double n=norm(a),s=n>=1e-20?1/n:0;return {a[0]*s,a[1]*s,a[2]*s};}
inline V mv(const M& a,V b){return {a[1]*b[1]+a[0]*b[0]+a[2]*b[2],a[3]*b[0]+a[4]*b[1]+a[5]*b[2],a[7]*b[1]+a[6]*b[0]+a[8]*b[2]};}
inline M mm(const M& a,const M& b){M c{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)c[i*3+j]=a[i*3]*b[j]+a[i*3+1]*b[3+j]+b[6+j]*a[i*3+2];return c;}
inline M generator(unsigned k){M g{};if(k==0){g[5]=-1;g[7]=1;}else if(k==1){g[2]=1;g[6]=-1;}else{g[1]=-1;g[3]=1;}return g;}
inline double point_scale(V p){double a=norm(p)/1000.0;return a/std::atan(a);}
struct Pose {std::array<std::byte,120> position{};M rotation{};bool rotation_fixed{};};
inline unsigned position_dof(const Pose& p){return std::to_integer<unsigned>(p.position[8])?0:3-std::to_integer<unsigned>(p.position[9]);}
inline V center(const Pose& p){return get<V>(p.position.data(),96);}
inline V position_basis(const Pose& p,unsigned k){if(position_dof(p)==2)return get<V>(p.position.data(),48+24*k);V v{};v[k]=1;return v;}
inline void update(Pose& p,std::span<const double> delta){unsigned n=position_dof(p),i=0;auto c=center(p);for(unsigned k=0;k<n;++k){auto v=position_basis(p,k);for(unsigned j=0;j<3;++j)c[j]+=v[j]*delta[i];++i;}if(n==2){auto origin=get<V>(p.position.data(),16);auto v=normalized(sub(c,origin));double radius=get<double>(p.position.data(),40);for(unsigned j=0;j<3;++j)c[j]=origin[j]+v[j]*radius;set(p.position.data(),96,c);v=normalized(sub(c,origin));unsigned axis=std::abs(v[0])>std::abs(v[1]);if(std::abs(v[axis])>std::abs(v[2]))axis=2;V e{};e[axis]=1;auto b=normalized(cross(e,v));auto a=cross(v,b);set(p.position.data(),48,a);set(p.position.data(),72,b);}else if(n)set(p.position.data(),96,c);
 if(!p.rotation_fixed){for(unsigned k=0;k<3;++k){double s=std::sin(delta[i+k]),c0=std::cos(delta[i+k]);M m{1,0,0,0,1,0,0,0,1};if(k==0){m[4]=c0;m[5]=-s;m[7]=s;m[8]=c0;}else if(k==1){m[0]=c0;m[2]=s;m[6]=-s;m[8]=c0;}else{m[0]=c0;m[1]=-s;m[3]=s;m[4]=c0;}p.rotation=mm(p.rotation,m);}i+=3;}if(i!=delta.size())throw std::runtime_error("pose delta extent");}
struct Observation {unsigned camera{},track{};std::uint64_t auxiliary{};float u{},v{},depth{},size{};};
static_assert(sizeof(Observation)==32);
struct Evaluation {std::array<double,2> residual{};std::vector<std::array<double,2>> jacobian;double cost{};};
constexpr std::array<unsigned,8> fitted{0,1,2,6,7,8,10,11};
inline Evaluation evaluate(const std::byte* calibration,const Pose& pose,V point,const Observation& o,bool derivatives=true,std::span<const unsigned> parameters=fitted){
 auto local=sub(point,center(pose));auto xyz=mv(pose.rotation,local);auto d=calibration::evaluate(calibration,xyz.data());double weight=o.size!=0?1.0/o.size:1.0;
 double f=get<double>(calibration,24),b1=get<double>(calibration,48),b2=get<double>(calibration,56),x=d.parameter[0][0],y=d.parameter[0][1];
 double u=(x*b1+x*f+y*b2)+(0.5*get<std::uint64_t>(calibration,8)+get<double>(calibration,32));
 double v=y*f+(0.5*get<std::uint64_t>(calibration,16)+get<double>(calibration,40));
 Evaluation e;e.residual={(o.u-u)*weight,(o.v-v)*weight};e.cost=(e.residual[0]*e.residual[0]+e.residual[1]*e.residual[1])*0.5;if(!derivatives)return e;
 for(auto k:parameters)e.jacobian.push_back({d.parameter[k][0]*weight,d.parameter[k][1]*weight});
 auto project_derivative=[&](V v){return std::array<double,2>{(d.point[0]*v[0]+d.point[1]*v[1]+d.point[2]*v[2])*weight,(d.point[3]*v[0]+d.point[4]*v[1]+d.point[5]*v[2])*weight};};
 for(unsigned k=0;k<position_dof(pose);++k){auto v=mv(pose.rotation,position_basis(pose,k));for(auto& x:v)x=-x;e.jacobian.push_back(project_derivative(v));}
 if(!pose.rotation_fixed)for(unsigned k=0;k<3;++k)e.jacobian.push_back(project_derivative(mv(mm(pose.rotation,generator(k)),local)));
 double scale=point_scale(point);for(unsigned k=0;k<3;++k){V v{pose.rotation[k]*scale,pose.rotation[3+k]*scale,pose.rotation[6+k]*scale};e.jacobian.push_back(project_derivative(v));}
 return e;
}
}
