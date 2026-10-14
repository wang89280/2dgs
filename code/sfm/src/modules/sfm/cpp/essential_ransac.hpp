#pragma once
#include "evidence.hpp"
#include "sturm_roots.hpp"
#include "relative_pose.hpp"
namespace essential {
using namespace ms;using U=std::uint64_t;using Ray=std::array<double,3>;using Matrix=std::array<double,9>;
struct Sample{std::array<double,15> first,second;U count;std::array<double,90> matrices;};static_assert(sizeof(Sample)==968);
// 143AB2810 uses two angular epipolar inequalities. Thresholds here are already squared.
bool inlier(const Matrix& m,Ray a,Ray b,double first,double second){
 double x=m[0]*a[0]+m[1]*a[1]+m[2]*a[2],y=m[3]*a[0]+m[4]*a[1]+m[5]*a[2],z=m[6]*a[0]+m[7]*a[1]+m[8]*a[2];
 double norm=(b[0]*b[0]+b[1]*b[1]+b[2]*b[2])*(y*y+x*x+z*z),dot=b[0]*x+b[1]*y+z*b[2];
 double tx=b[0]*m[0]+b[1]*m[3]+b[2]*m[6],ty=b[0]*m[1]+b[1]*m[4]+b[2]*m[7],tz=b[0]*m[2]+b[1]*m[5]+b[2]*m[8];
 double other=a[0]*tx+a[1]*ty+a[2]*tz;
 return norm*second>dot*dot&&(a[1]*a[1]+a[0]*a[0]+a[2]*a[2])*(ty*ty+tx*tx+tz*tz)*first>other*other;
}
// 143AB1480 evaluates unsquared threshold vectors, with its own operation order.
bool score_inlier(const Matrix& m,Ray a,Ray b,double first,double second){
 double x=m[1]*a[1]+m[0]*a[0]+m[2]*a[2],y=m[3]*a[0]+m[4]*a[1]+m[5]*a[2],z=m[6]*a[0]+m[7]*a[1]+m[8]*a[2];
 double norm=(b[1]*b[1]+b[0]*b[0]+b[2]*b[2])*(y*y+x*x+z*z),dot=x*b[0]+b[1]*y+b[2]*z;
 double tx=b[1]*m[3]+b[0]*m[0]+b[2]*m[6],ty=b[0]*m[1]+b[1]*m[4]+b[2]*m[7],tz=b[0]*m[2]+b[1]*m[5]+b[2]*m[8];
 double other=tx*a[0]+ty*a[1]+tz*a[2];double other_norm=(ty*ty+tx*tx+tz*tz)*(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
 return second*second*norm>dot*dot&&first*first*other_norm>other*other;
}
int iteration_limit(int current,double ratio,double confidence){if(!(confidence>0.0&&confidence<1.0))return current;double n=std::log(1.0-confidence)/std::log(1.0-std::pow(1.0-(1.0-ratio),5.0));if(current<=n)return current;return (std::max)(10,int(n));}

struct Estimation {std::vector<U> best;std::vector<Matrix> matrices;std::vector<std::array<double,30>> generated_samples;int iterations;};
inline Estimation estimate(std::span<const Ray> a,std::span<const Ray> b,std::span<const double> ta,std::span<const double> tb,double confidence,std::span<const Sample> samples={},bool independent=true){
 if(a.size()!=b.size()||a.size()<6||ta.empty()||ta.size()!=tb.size())throw std::runtime_error("invalid essential input");
  std::vector<U> best(ta.size());std::vector<Matrix> matrices(ta.size());std::vector<std::array<double,30>> generated_samples;
  U state=1;int limit=500,iterations=0;
  while(iterations<limit){
   if(!independent&&unsigned(iterations)>=samples.size())throw std::runtime_error("native five-point trace exhausted before independently computed stop");
   std::array<unsigned,5> ids;for(unsigned k=0;k<5;++k){unsigned retries=0;for(;;){state=16807*state%2147483647;ids[k]=unsigned(state%a.size());if(std::find(ids.begin(),ids.begin()+k,ids[k])==ids.begin()+k)break;if(++retries>=1000)throw std::runtime_error("sample duplicate retry limit");}}
   std::array<double,30> generated;for(unsigned k=0;k<5;++k)for(unsigned j=0;j<3;++j){generated[3*k+j]=a[ids[k]][j];generated[15+3*k+j]=b[ids[k]][j];}generated_samples.push_back(generated);
   std::vector<Matrix> candidates;if(independent)candidates=five::solve(generated.data(),generated.data()+15);else{auto&sample=samples[iterations];if(sample.count<=10)for(U c=0;c<sample.count;++c){Matrix m;std::copy_n(sample.matrices.begin()+9*c,9,m.begin());candidates.push_back(m);}}++iterations;
   for(const Matrix& m:candidates){int updated_limit=-1;
    for(unsigned t=0;t<ta.size();++t){U count=0;for(std::size_t i=0;i<a.size();++i)count+=score_inlier(m,a[i],b[i],ta[t],tb[t]);if(count>best[t]){best[t]=count;matrices[t]=m;double ratio=double(count)/double(a.size());int next=ratio>=1.0?0:iteration_limit(limit,ratio,confidence);updated_limit=(std::max)(updated_limit,next);}}
    if(updated_limit>=0)limit=updated_limit;
   }
  }

 return {std::move(best),std::move(matrices),std::move(generated_samples),iterations};
}
// 143AB21F0: compare every narrower level to HALF THE ORIGINAL widest count.
// This wrapper passes the selected raw threshold to 143AB2810; its mask differs
// from the squared-threshold scoring predicate in 143AB1480. Preserve that ABI.
inline unsigned select_level(std::span<const U> counts){
 if(counts.size()!=10)throw std::runtime_error("native wrapper requires ten thresholds");
 unsigned selected=0;if(counts[0]>64)for(unsigned i=1;i<10;++i){if(double(counts[i])*2.0<double(counts[0]))break;selected=i;}return selected;
}
inline std::vector<unsigned char> selected_mask(const Estimation&r,std::span<const Ray>a,std::span<const Ray>b,std::span<const double>ta,std::span<const double>tb){
 unsigned t=select_level(r.best);std::vector<unsigned char>mask(a.size());if(r.best[t])for(std::size_t i=0;i<a.size();++i)mask[i]=inlier(r.matrices[t],a[i],b[i],ta[t],tb[t]);return mask;
}
}
