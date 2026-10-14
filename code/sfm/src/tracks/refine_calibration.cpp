#define MS_SFM_NO_MAIN
#include "../modules/sfm/cpp/independent_sfm.cpp"

namespace calibrated {
struct Policy {
 std::array<std::byte,256> target{};
 std::vector<unsigned> parameters;
 double mass=.02;
 // Track maintenance uses a finite feasibility bound; reconstruction uses the unbounded solver.
 double observation_gate=INFINITY;
 double sigma(unsigned bit)const {
  if(bit==0)return .02*get<double>(target.data(),24);
  if(bit==1||bit==2)return 10;
  if(bit==6)return .03;
  if(bit==7)return .05;
  if(bit==10||bit==11)return .001;
  throw std::runtime_error("unsupported constrained intrinsic");
 }
 unsigned offset(unsigned bit)const{return 24+8*bit+(bit>=6?8:0);}
 IntrinsicPrior prior(unsigned count)const {
  IntrinsicPrior p{};std::memcpy(p.data()+8,target.data(),256);
  for(auto bit:parameters)set(p.data(),264+offset(bit),std::sqrt(count*mass)/sigma(bit));
  return p;
 }
 bool allowed(const State&s)const {
  if(!(get<double>(s.calibration.data(),24)>0))return false;
  for(auto bit:parameters){double z=(get<double>(s.calibration.data(),offset(bit))-get<double>(target.data(),offset(bit)))/sigma(bit);if(!std::isfinite(z)||std::abs(z)>3)return false;}
  return true;
 }
};

double objective(const State&s,const Data&d){
 double value=sfm::huber_objective(s,d);
 for(const auto&p:d.priors)value+=evaluate_prior(s.calibration.data(),p,d.parameters).cost;
 return value;
}

bool feasible_observations(const State&s,const Data&d,const Policy&p){
 if(!std::isfinite(p.observation_gate))return true;
 for(auto o:d.observations){o.size=1;const auto&pose=s.poses[o.camera];const auto&x=s.points[o.track];
  double z=mv(pose.rotation,sub(x,center(pose)))[2];
  double e=sfm::pixel_error(s.calibration,pose,x,o);
  if(!(z>1e-10)||!std::isfinite(e)||e>p.observation_gate)return false;
 }return true;
}

void optimize_bounded(State&s,Data&d,const Policy&policy,std::ostream&log){
 sfm::configure(d,s);double lambda=.001,current=cost(s,d);
 for(unsigned it=0;it<25;++it){
  try{
   auto normal=assemble(s,d,lambda);auto delta=solve(normal.schur);auto dp=back_substitute(normal,delta);
   double direction=std::inner_product(delta.begin(),delta.end(),normal.gradient.begin(),0.0);
   for(unsigned i=0;i<dp.size();++i)for(unsigned k=0;k<3;++k)direction+=dp[i][k]*normal.points[i].gradient[k];
   double alpha=1,next=INFINITY;State candidate;
   while(alpha>1e-10){candidate=trial(s,d,delta,dp,alpha);next=policy.allowed(candidate)&&feasible_observations(candidate,d,policy)?cost(candidate,d):INFINITY;if(std::isfinite(next)&&next<current&&next<=current-alpha*.001*direction)break;alpha*=.5;}
   log<<"{\"iteration\":"<<it<<",\"cost\":"<<current<<",\"alpha\":"<<alpha<<",\"accepted\":"<<(alpha>1e-10?"true":"false")<<"}\n";log.flush();
   if(alpha<=1e-10){lambda*=10;if(lambda>1e6)break;continue;}
   double improvement=current-next;s=std::move(candidate);current=next;lambda=std::max(1e-10,lambda/10);
   if(improvement<1e-9*std::max(1.0,current))break;
  }catch(const std::runtime_error&e){lambda*=10;log<<"{\"linear_failure\":true,\"lambda\":"<<lambda<<"}\n";if(lambda>1e6)throw;}
 }
}

sfm::Reconstruction load_model(const fs::path&input){
 sfm::Reconstruction r;auto cal=read_bytes(input/"calibration.bin");if(cal.size()!=256)throw std::runtime_error("calibration size");std::copy(cal.begin(),cal.end(),r.s.calibration.begin());
 auto ids=read_array<unsigned>(input/"camera_ids.bin");auto values=read_array<std::array<double,12>>(input/"poses.bin");if(ids.size()!=values.size()||ids.size()<3)throw std::runtime_error("camera count");
 for(unsigned i=0;i<ids.size();++i){Pose p;std::copy(values[i].begin()+3,values[i].end(),p.rotation.begin());set(p.position.data(),96,V{values[i][0],values[i][1],values[i][2]});sfm::add(r,ids[i],p);}
 r.s.points=read_array<V>(input/"points.bin");auto active=read_array<unsigned>(input/"active_points.bin");for(auto id:active){auto&t=r.tracks[id];t.xyz=r.s.points.at(id);t.valid=true;}
 for(auto o:read_array<sfm::Obs>(input/"observations.bin")){o.camera=ids.at(o.camera);o.size=1;r.tracks.at(o.track).obs.push_back(o);}
 return r;
}

void refine(sfm::Reconstruction&r,const Policy&policy,std::ostream&log){
 sfm::configure_gauge(r);auto d=sfm::collect(r);d.parameters=policy.parameters;
 if(!d.parameters.empty())d.priors={policy.prior(unsigned(d.observations.size()))};
 for(unsigned irls=0;irls<5;++irls){auto before=r.s;double previous=objective(r.s,d);
  for(auto&o:d.observations){auto unweighted=o;unweighted.size=1;double e=sfm::pixel_error(r.s.calibration,r.s.poses[o.camera],r.s.points[o.track],unweighted);o.size=float(std::sqrt(std::max(1.0,e/2)));}
  optimize_bounded(r.s,d,policy,log);double next=objective(r.s,d);bool accept=std::isfinite(next)&&next<=previous+1e-8*std::max(1.0,previous)&&policy.allowed(r.s);
  log<<"{\"irls\":"<<irls<<",\"huber_prior_before\":"<<previous<<",\"huber_prior_after\":"<<next<<",\"accepted\":"<<(accept?"true":"false")<<"}\n";log.flush();
  if(!accept){r.s=std::move(before);break;}if(previous-next<1e-7*std::max(1.0,previous))break;
 }
 for(auto id:d.active)r.tracks.at(id).xyz=r.s.points[id];
}

void self_test(){
 State s;initial_calibration(s,3072,2048);Policy p;p.target=s.calibration;p.parameters={0,1,2,6,7,10,11};auto prior=p.prior(10000);
 for(unsigned i=0;i<p.parameters.size();++i){auto bit=p.parameters[i],off=p.offset(bit);double mean=get<double>(p.target.data(),off),sigma=p.sigma(bit);State q=s;set(q.calibration.data(),off,mean+.7*sigma);auto e=evaluate_prior(q.calibration.data(),prior,p.parameters);
  if(std::abs(e.cost-49)>1e-7)throw std::runtime_error("prior cost scaling");
  double h=sigma*1e-4;auto a=q,b=q;set(a.calibration.data(),off,mean+.7*sigma+h);set(b.calibration.data(),off,mean+.7*sigma-h);double numeric=(evaluate_prior(a.calibration.data(),prior,p.parameters).cost-evaluate_prior(b.calibration.data(),prior,p.parameters).cost)/(2*h);double analytic=-e.residual[i]*e.diagonal[i];if(std::abs(numeric-analytic)>1e-5*std::max(1.0,std::abs(analytic)))throw std::runtime_error("prior gradient sign");
  if(!p.allowed(q))throw std::runtime_error("valid bounds rejected");set(q.calibration.data(),off,mean+3.01*sigma);if(p.allowed(q))throw std::runtime_error("invalid bounds accepted");
 }
 // Synthetic camera fixture for an independent numerical intrinsic Jacobian check.
 initial_calibration(s,3072,2048);Policy q;q.target=s.calibration;q.parameters={0};q.mass=0;Data d;d.parameters={0};
 for(unsigned cam=0;cam<4;++cam){Pose pose;pose.rotation=identity;pose.rotation_fixed=true;pose.position[8]=std::byte{1};set(pose.position.data(),96,V{-.4+.27*cam,.03*cam,0});s.poses.push_back(pose);}
 for(unsigned k=0;k<100;++k){V point{(int(k%10)-4.5)*.09,(int(k/10)-4.5)*.08,2.5+.03*(k%7)};s.points.push_back(point);d.active.push_back(k);for(unsigned cam=0;cam<4;++cam){sfm::Obs o{};o.camera=cam;o.track=k;o.size=1;auto z=evaluate(s.calibration.data(),s.poses[cam],point,o,false).residual;o.u=float(-z[0]);o.v=float(-z[1]);if(sfm::pixel_error(s.calibration,s.poses[cam],point,o)>1){o.u=-o.u;o.v=-o.v;}d.observations.push_back(o);}}
 // Validate analytic intrinsic Jacobians against finite differences, independently of optimizer convergence.
 q.parameters={0,1,2,6,7,10,11};auto obs=d.observations[7];auto analytic=evaluate(s.calibration.data(),s.poses[obs.camera],s.points[obs.track],obs,true,q.parameters);
 for(unsigned j=0;j<q.parameters.size();++j){unsigned bit=q.parameters[j],off=q.offset(bit);double original=get<double>(s.calibration.data(),off),h=q.sigma(bit)*1e-4;auto a=s,b=s;set(a.calibration.data(),off,original+h);set(b.calibration.data(),off,original-h);auto ea=evaluate(a.calibration.data(),a.poses[obs.camera],a.points[obs.track],obs,false),eb=evaluate(b.calibration.data(),b.poses[obs.camera],b.points[obs.track],obs,false);
  for(unsigned k=0;k<2;++k)if(std::abs((ea.residual[k]-eb.residual[k])/(2*h)+analytic.jacobian[j][k])>1e-5)throw std::runtime_error("intrinsic Jacobian convention");
 }
 std::cout<<"PASS prior scale, seven-parameter prior gradients, hard bounds and seven intrinsic Jacobians\n";
}
}

