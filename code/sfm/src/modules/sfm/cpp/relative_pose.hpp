#pragma once
#include "triangulate.hpp"
namespace ms::five {
using Projection=std::array<double,12>;
inline constexpr Projection identity_projection{1,0,0,0,0,1,0,0,0,0,1,0};
inline double det3(const std::array<double,9>&m){return (m[4]*m[8]-m[5]*m[7])*m[0]-(m[3]*m[8]-m[5]*m[6])*m[1]+(m[3]*m[7]-m[4]*m[6])*m[2];}
inline std::array<Projection,4> projection_candidates(const std::array<double,9>&e){auto svd=square_svd<3>(e);if(det3(svd.ut)<0)for(auto&x:svd.ut)x=-x;if(det3(svd.vt)<0)for(auto&x:svd.vt)x=-x;auto&u=svd.ut;auto&v=svd.vt;std::array<Projection,4> result{};
 for(int branch=0;branch<2;++branch)for(int sign=0;sign<2;++sign){auto&p=result[branch*2+sign];for(int r=0;r<3;++r){for(int c=0;c<3;++c)p[r*4+c]=branch?(u[r]*v[3+c]-u[3+r]*v[c])+u[6+r]*v[6+c]:(u[3+r]*v[c]-u[r]*v[3+c])+u[6+r]*v[6+c];p[r*4+3]=sign?-u[6+r]:u[6+r];}}
 return result;
}
struct PoseResult {Projection first=identity_projection,second;std::array<std::size_t,4> positive{};int selected=0;};
inline PoseResult relative_pose(const std::array<double,9>&e,std::span<const std::array<double,3>>a,std::span<const std::array<double,3>>b,std::span<const unsigned char>mask={}){if(a.size()!=b.size()||(!mask.empty()&&mask.size()!=a.size()))throw std::runtime_error("ray size mismatch");auto candidates=projection_candidates(e);PoseResult result;
 for(int i=0;i<a.size();++i)if(mask.empty()||mask[i])for(int k=0;k<4;++k){auto point=triangulate(identity_projection.data(),candidates[k].data(),a[i].data(),b[i].data());double inv=1/point[3];for(auto&x:point)x=x*inv;auto forward=[&](const Projection&p,const std::array<double,3>&ray){double q[3];for(int r=0;r<3;++r)q[r]=p[r*4]*point[0]+p[r*4+1]*point[1]+p[r*4+2]*point[2]+p[r*4+3]*point[3];return q[0]*ray[0]+q[1]*ray[1]+q[2]*ray[2]>0;};if(forward(identity_projection,a[i])&&forward(candidates[k],b[i]))++result.positive[k];}
 for(int i=1;i<4;++i)if(result.positive[i]>result.positive[result.selected])result.selected=i;result.second=candidates[result.selected];return result;
}
}
