#define GRAPH6_NO_MAIN
#include "refine_graph.cpp"

// Image-derived directional information, fixed during all geometry updates.
// Original UV, track membership and calibration remain unchanged.
namespace covariance_ba {
using W=std::array<double,4>;
using Key=std::pair<unsigned,unsigned>;
std::array<double,2> whiten(W w,std::array<double,2> v){return {w[0]*v[0]+w[1]*v[1],w[2]*v[0]+w[3]*v[1]};}
auto evaluation(const State&s,const Data&d,unsigned oi,W w,bool jac){
 const auto&o=d.observations[oi];auto e=bundle::evaluate(s.calibration.data(),s.poses[o.camera],s.points[o.track],o,jac,d.parameters);
 e.residual=whiten(w,e.residual);for(auto&j:e.jacobian)j=whiten(w,j);
 e.cost=.5*(e.residual[0]*e.residual[0]+e.residual[1]*e.residual[1]);return e;
}
Normal assemble(const State&s,const Data&d,std::span<const W>w,double lambda){
 Normal normal;auto&S=normal.schur;S.n=d.n;S.a.resize(d.n*d.n);S.pattern.resize(d.n*d.n);S.b.resize(d.n);normal.gradient.resize(d.n);
 for(unsigned i=0;i<d.n;++i)S.pattern[i*d.n+i]=1;
 for(auto point:d.active){PointNormal p;std::map<unsigned,V>cross;
  for(auto oi:d.by_point[point]){const auto&o=d.observations[oi];auto e=evaluation(s,d,oi,w[oi],true);normal.cost+=e.cost;++normal.observations;
   unsigned n=unsigned(e.jacobian.size())-3;std::vector<unsigned>ids(n);auto k=d.parameters.size();std::iota(ids.begin(),ids.begin()+k,0);std::iota(ids.begin()+k,ids.end(),d.starts[o.camera]);
   for(unsigned i=0;i<n;++i){auto a=e.jacobian[i];normal.gradient[ids[i]]+=a[0]*e.residual[0]+a[1]*e.residual[1];
    for(unsigned j=0;j<n;++j){auto b=e.jacobian[j];S.a[ids[i]*d.n+ids[j]]+=a[0]*b[0]+a[1]*b[1];}
    auto&z=cross[ids[i]];for(unsigned j=0;j<3;++j)z[j]+=a[0]*e.jacobian[n+j][0]+a[1]*e.jacobian[n+j][1];}
   for(unsigned i=0;i<3;++i){auto a=e.jacobian[n+i];p.gradient[i]+=a[0]*e.residual[0]+a[1]*e.residual[1];for(unsigned j=0;j<3;++j)p.v[i*3+j]+=a[0]*e.jacobian[n+j][0]+a[1]*e.jacobian[n+j][1];}}
  for(unsigned i=0;i<3;++i)p.v[i*3+i]*=1+lambda;p.inverse=inverse3(p.v);
  for(auto&[id,z]:cross){p.ids.push_back(id);p.cross.push_back(z);}normal.points.push_back(std::move(p));}
 if(!d.priors.empty()||!d.parameters.empty())throw std::runtime_error("covariance experiment requires fixed K");
 S.b=normal.gradient;for(unsigned i=0;i<d.n;++i)S.a[i*d.n+i]*=1+lambda;
 for(auto&p:normal.points){auto b=mv(p.inverse,p.gradient);for(unsigned i=0;i<p.ids.size();++i){auto z=p.cross[i];S.b[p.ids[i]]-=z[0]*b[0]+z[1]*b[1]+z[2]*b[2];auto v=mv(p.inverse,z);for(unsigned j=0;j<p.ids.size();++j){auto t=p.cross[j];S.pattern[p.ids[i]*d.n+p.ids[j]]=1;S.a[p.ids[i]*d.n+p.ids[j]]-=v[0]*t[0]+v[1]*t[1]+v[2]*t[2];}}}return normal;
}
double cost(const State&s,const Data&d,std::span<const W>w){double c=0;for(unsigned i=0;i<w.size();++i)c+=evaluation(s,d,i,w[i],false).cost;return c;}
double objective(const State&s,Data d,std::span<const W>w){double c=0;for(auto&o:d.observations)o.size=1;for(unsigned i=0;i<w.size();++i){auto e=evaluation(s,d,i,w[i],false);double r=std::sqrt(2*e.cost);c+=r<=2?.5*r*r:2*(r-1);}return c;}
void optimize(State&s,Data&d,std::span<const W>w,std::ostream&log){
 sfm::configure(d,s);double lambda=.001,current=cost(s,d,w);calibrated::Policy p;p.target=s.calibration;p.observation_gate=1e6;
 for(unsigned it=0;it<25;++it){auto n=assemble(s,d,w,lambda);auto delta=solve(n.schur);auto dp=back_substitute(n,delta);
  double direction=std::inner_product(delta.begin(),delta.end(),n.gradient.begin(),0.0);for(unsigned i=0;i<dp.size();++i)for(unsigned k=0;k<3;++k)direction+=dp[i][k]*n.points[i].gradient[k];
  double alpha=1,next=INFINITY;State candidate;
  while(alpha>1e-10){candidate=trial(s,d,delta,dp,alpha);next=calibrated::feasible_observations(candidate,d,p)?cost(candidate,d,w):INFINITY;if(std::isfinite(next)&&next<current&&next<=current-alpha*.001*direction)break;alpha*=.5;}
  log<<"{\"iteration\":"<<it<<",\"cost\":"<<current<<",\"alpha\":"<<alpha<<"}\n";log.flush();
  if(alpha<=1e-10){lambda*=10;if(lambda>1e6)break;continue;}double gain=current-next;s=std::move(candidate);current=next;lambda=std::max(1e-10,lambda/10);if(gain<1e-9*std::max(1.,current))break;
 }
}
void refine(sfm::Reconstruction&r,const std::map<Key,W>&weights,std::ostream&log){
 sfm::configure_gauge(r);auto d=sfm::collect(r);std::vector<W>w;for(auto o:d.observations)w.push_back(weights.at({r.ids[o.camera],o.track}));
 for(unsigned irls=0;irls<5;++irls){auto before=r.s;double previous=objective(r.s,d,w);
  for(unsigned i=0;i<w.size();++i){d.observations[i].size=1;double e=std::sqrt(2*evaluation(r.s,d,i,w[i],false).cost);d.observations[i].size=float(std::sqrt(std::max(1.,e/2)));}
  optimize(r.s,d,w,log);double next=objective(r.s,d,w);bool accept=std::isfinite(next)&&next<=previous+1e-8*std::max(1.,previous);
  log<<"{\"irls\":"<<irls<<",\"before\":"<<previous<<",\"after\":"<<next<<",\"accepted\":"<<(accept?"true":"false")<<"}\n";log.flush();
  if(!accept){r.s=before;break;}if(previous-next<1e-7*std::max(1.,previous))break;
 }for(auto id:d.active)r.tracks.at(id).xyz=r.s.points[id];
}
void self_test(){
 State s;initial_calibration(s,3072,2048);Data d;d.parameters.clear();
 for(unsigned i=0;i<3;++i){Pose p;p.rotation=identity;set(p.position.data(),96,V{.2*i,0,0});if(i==0){p.rotation_fixed=true;p.position[8]=std::byte{1};}s.poses.push_back(p);}
 for(unsigned t=0;t<8;++t){s.points.push_back({.1*t,.07*(t%3),3.+.1*t});d.active.push_back(t);for(unsigned i=0;i<3;++i){sfm::Obs o{};o.camera=i;o.track=t;o.size=1;auto e=bundle::evaluate(s.calibration.data(),s.poses[i],s.points[t],o,false);o.u=float(-e.residual[0]+.7);o.v=float(-e.residual[1]-1.5);d.observations.push_back(o);}}
 sfm::configure(d,s);std::vector<W>unit(d.observations.size(),W{1,0,0,1});auto a=::assemble(s,d,.01),b=assemble(s,d,unit,.01);
 for(unsigned i=0;i<a.schur.a.size();++i)if(a.schur.a[i]!=b.schur.a[i])throw std::runtime_error("unit Schur mismatch");
 for(unsigned i=0;i<a.schur.b.size();++i)if(a.schur.b[i]!=b.schur.b[i])throw std::runtime_error("unit RHS mismatch");
 if(std::abs(objective(s,d,unit)-sfm::huber_objective(s,d))>1e-10)throw std::runtime_error("unit objective mismatch");
 std::vector<W>w(d.observations.size(),W{1.1,.3,.3,.7});auto irls=d;
 for(unsigned i=0;i<w.size();++i){double e=std::sqrt(2*evaluation(s,d,i,w[i],false).cost);irls.observations[i].size=float(std::sqrt(std::max(1.,e/2)));}
 auto normal=assemble(s,irls,w,.001);double h=1e-6;
 for(unsigned k=0;k<3;++k){auto plus=s,minus=s;plus.points[2][k]+=h;minus.points[2][k]-=h;
  double num=(objective(plus,d,w)-objective(minus,d,w))/(2*h),ana=-normal.points[2].gradient[k]/point_scale(s.points[2]);
  if(std::abs(num-ana)>1e-5*std::max(1.,std::abs(num)))throw std::runtime_error("anisotropic robust gradient mismatch");}
 std::cout<<"PASS unit Schur/RHS equivalence and anisotropic robust numerical XYZ gradients\n";
}
}
int wmain(int argc,wchar_t**argv){try{
 if(argc==2&&std::wstring(argv[1])==L"--self-test"){covariance_ba::self_test();return 0;}
 if(argc!=4||fs::exists(argv[2]))throw std::runtime_error("usage: refine_covariance INPUT_SFM NEW_OUTPUT WHITENING_BIN");
 auto r=calibrated::load_model(argv[1]);auto cal=r.s.calibration;auto obs=read_array<sfm::Obs>(fs::path(argv[1])/"observations.bin");auto w=read_array<covariance_ba::W>(argv[3]);
 if(w.size()!=obs.size())throw std::runtime_error("weight extent");std::map<covariance_ba::Key,covariance_ba::W>weights;
 for(unsigned i=0;i<w.size();++i){auto a=w[i];if(!std::all_of(a.begin(),a.end(),[](double v){return std::isfinite(v);})||std::abs(a[1]-a[2])>1e-8||a[0]<=0||a[3]<=0||a[0]*a[3]-a[1]*a[2]<=0)throw std::runtime_error("non-SPD whitening");if(!weights.emplace(covariance_ba::Key{r.ids.at(obs[i].camera),obs[i].track},a).second)throw std::runtime_error("duplicate observation");}
 fs::path out=argv[2];fs::create_directories(out);std::ofstream log(out/"optimization.jsonl");log<<std::setprecision(17);
 for(unsigned round=0;round<3;++round){log<<"{\"round\":"<<round<<",\"ba_call\":true}\n";covariance_ba::refine(r,weights,log);}
 if(r.s.calibration!=cal)throw std::runtime_error("intrinsics changed");graph6::save_model(r,argv[1],out);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
