#define GRAPH6_NO_MAIN
#include "refine_graph.cpp"

// Camera-center ring prior only: orientations/angles remain free. This is a
// turntable hypothesis, not a universal property of SfM image collections.
namespace orbit {
struct Policy {V origin{},axis{};double radius{},weight{};};
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
struct E {std::array<double,2> residual{};std::vector<std::array<double,2>> jacobian;};
E evaluate(const Pose&pose,const Policy&p){
 auto v=sub(center(pose),p.origin);double z=dot(v,p.axis);V radial{};for(unsigned k=0;k<3;++k)radial[k]=v[k]-z*p.axis[k];
 double radius=norm(radial);if(radius<1e-8)throw std::runtime_error("camera at orbit axis");
 E e;e.residual={-z*p.weight,(p.radius-radius)*p.weight};
 for(unsigned k=0;k<position_dof(pose);++k){auto b=position_basis(pose,k);e.jacobian.push_back({dot(b,p.axis)*p.weight,dot(b,radial)/radius*p.weight});}return e;
}
double penalty(const State&s,const Policy&p){double cost=0;for(const auto&pose:s.poses){auto e=evaluate(pose,p);for(double r:e.residual)cost+=.5*r*r;}return cost;}
void add(Normal&n,const State&s,const Data&d,const Policy&p,double lambda){
 for(unsigned cam=0;cam<s.poses.size();++cam){auto e=evaluate(s.poses[cam],p);for(unsigned i=0;i<e.jacobian.size();++i){auto a=e.jacobian[i];unsigned id=d.starts[cam]+i;double g=a[0]*e.residual[0]+a[1]*e.residual[1];n.gradient[id]+=g;n.schur.b[id]+=g;
  for(unsigned j=0;j<e.jacobian.size();++j){auto b=e.jacobian[j];unsigned jd=d.starts[cam]+j;n.schur.a[id*d.n+jd]+=(a[0]*b[0]+a[1]*b[1])*(i==j?1+lambda:1);n.schur.pattern[id*d.n+jd]=1;}
 }}
}
void optimize(State&s,Data&d,const Policy&p,std::ostream&log){
 sfm::configure(d,s);double lambda=.001,current=cost(s,d)+penalty(s,p);calibrated::Policy feasible;feasible.target=s.calibration;feasible.observation_gate=1e6;
 for(unsigned it=0;it<25;++it){auto n=assemble(s,d,lambda);add(n,s,d,p,lambda);auto delta=solve(n.schur);auto dp=back_substitute(n,delta);
  double direction=std::inner_product(delta.begin(),delta.end(),n.gradient.begin(),0.0);for(unsigned i=0;i<dp.size();++i)for(unsigned k=0;k<3;++k)direction+=dp[i][k]*n.points[i].gradient[k];
  double alpha=1,next=INFINITY;State candidate;
  while(alpha>1e-10){candidate=trial(s,d,delta,dp,alpha);next=calibrated::feasible_observations(candidate,d,feasible)?cost(candidate,d)+penalty(candidate,p):INFINITY;if(std::isfinite(next)&&next<current&&next<=current-alpha*.001*direction)break;alpha*=.5;}
  log<<"{\"iteration\":"<<it<<",\"cost\":"<<current<<",\"alpha\":"<<alpha<<"}\n";log.flush();
  if(alpha<=1e-10){lambda*=10;if(lambda>1e6)break;continue;}double improvement=current-next;s=std::move(candidate);current=next;lambda=std::max(1e-10,lambda/10);if(improvement<1e-9*std::max(1.,current))break;
 }
}
void refine(sfm::Reconstruction&r,const Policy&p,std::ostream&log){
 sfm::configure_gauge(r);auto d=sfm::collect(r);
 for(unsigned i=0;i<5;++i){auto before=r.s;double previous=sfm::huber_objective(r.s,d)+penalty(r.s,p);
  for(auto&o:d.observations){o.size=1;double e=sfm::pixel_error(r.s.calibration,r.s.poses[o.camera],r.s.points[o.track],o);o.size=float(std::sqrt(std::max(1.,e/2)));}
  optimize(r.s,d,p,log);double after=sfm::huber_objective(r.s,d)+penalty(r.s,p);bool accepted=std::isfinite(after)&&after<=previous+1e-8*std::max(1.,previous);
  log<<"{\"irls\":"<<i<<",\"before\":"<<previous<<",\"after\":"<<after<<",\"accepted\":"<<(accepted?"true":"false")<<"}\n";
  if(!accepted){r.s=before;break;}if(previous-after<1e-7*std::max(1.,previous))break;
 }for(auto id:d.active)r.tracks.at(id).xyz=r.s.points[id];
}
void test(){Policy p{{.1,.2,.3},normalized(V{.1,.3,1}),4,2};Pose pose;pose.rotation=identity;set(pose.position.data(),96,V{3,2,1});
 for(unsigned mode=0;mode<2;++mode){if(mode){pose.position[9]=std::byte{1};set(pose.position.data(),16,V{});set(pose.position.data(),40,norm(center(pose)));auto n=normalized(center(pose)),b=normalized(cross(n,V{0,0,1}));set(pose.position.data(),48,b);set(pose.position.data(),72,cross(n,b));}
  auto e=evaluate(pose,p);for(unsigned j=0;j<position_dof(pose);++j){auto a=pose,b=pose;std::vector<double>d(position_dof(pose)+3);d[j]=1e-6;update(a,d);d[j]=-1e-6;update(b,d);auto ea=evaluate(a,p),eb=evaluate(b,p);for(unsigned k=0;k<2;++k)if(std::abs((ea.residual[k]-eb.residual[k])/2e-6+e.jacobian[j][k])>1e-6)throw std::runtime_error("orbit prior derivative");}
 }std::cout<<"PASS orbit residual derivatives for free and scale-gauged cameras\n";
}
}
int wmain(int argc,wchar_t**argv){try{
 if(argc==2&&std::wstring(argv[1])==L"--self-test"){orbit::test();return 0;}
 if(argc!=4||fs::exists(argv[2]))throw std::runtime_error("usage: refine_orbit INPUT_SFM NEW_OUTPUT ORBIT_PRIOR_BIN");
 auto v=read_array<double>(argv[3]);if(v.size()!=8)throw std::runtime_error("orbit prior extent");
 orbit::Policy p{{v[0],v[1],v[2]},{v[3],v[4],v[5]},v[6],v[7]};if(std::abs(norm(p.axis)-1)>1e-8||p.radius<=0||p.weight<=0)throw std::runtime_error("invalid orbit prior");
 auto r=calibrated::load_model(argv[1]);auto original=r.s.calibration;fs::path out=argv[2];fs::create_directories(out);std::ofstream log(out/"optimization.jsonl");log<<std::setprecision(17);
 for(unsigned round=0;round<3;++round)orbit::refine(r,p,log);
 if(r.s.calibration!=original)throw std::runtime_error("fixed intrinsics changed");graph6::save_model(r,argv[1],out);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
