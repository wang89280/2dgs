#define GRAPH6_NO_MAIN
#include "refine_graph.cpp"
#ifndef IMAGE_SHIFT_LIMIT
#define IMAGE_SHIFT_LIMIT 2
#endif

// Fixed-intrinsic BA on independently remeasured image observations. This
// target never deletes, merges or invents observations. The robust objective
// explicitly retains the image-derived base sigma across IRLS iterations.
namespace image_ba {
using Identity=std::pair<unsigned,unsigned>; // original camera ID, track ID

double rho(double e){return e<=2?.5*e*e:2*(e-1);}
double objective(const State&s,const Data&d,std::span<const float>sigma){
 if(sigma.size()!=d.observations.size())throw std::runtime_error("sigma extent");
 double loss=0;for(unsigned i=0;i<sigma.size();++i){auto o=d.observations[i];
  double e=sfm::pixel_error(s.calibration,s.poses[o.camera],s.points[o.track],o)/sigma[i];loss+=rho(e);
 }return loss;
}

void refine(sfm::Reconstruction&r,const std::map<Identity,float>&sigmas,std::ostream&log){
 sfm::configure_gauge(r);auto d=sfm::collect(r);std::vector<float>sigma;
 for(auto o:d.observations)sigma.push_back(sigmas.at({r.ids[o.camera],o.track}));
 calibrated::Policy fixed;fixed.target=r.s.calibration;
 // Finite large gate enables the shared positive-depth check without trimming
 // residuals or rejecting independently measured coordinates just above 4px.
 fixed.observation_gate=1e6;
 for(unsigned irls=0;irls<5;++irls){auto before=r.s;double previous=objective(r.s,d,sigma);
  for(unsigned i=0;i<d.observations.size();++i){auto&o=d.observations[i];
   double normalized=sfm::pixel_error(r.s.calibration,r.s.poses[o.camera],r.s.points[o.track],o)/sigma[i];
   o.size=float(sigma[i]*std::sqrt(std::max(1.0,normalized/2)));
  }
  calibrated::optimize_bounded(r.s,d,fixed,log);
  double next=objective(r.s,d,sigma);bool accept=std::isfinite(next)&&next<=previous+1e-8*std::max(1.0,previous);
  log<<"{\"irls\":"<<irls<<",\"normalized_huber_before\":"<<previous<<",\"normalized_huber_after\":"<<next<<",\"accepted\":"<<(accept?"true":"false")<<"}\n";log.flush();
  if(!accept){r.s=std::move(before);break;}
  if(previous-next<1e-7*std::max(1.0,previous))break;
 }
 for(auto id:d.active)r.tracks.at(id).xyz=r.s.points[id];
}

void self_test(){
 State s;initial_calibration(s,3072,2048);Pose p;p.rotation=identity;s.poses.push_back(p);s.points.push_back(V{.1,.05,3});
 sfm::Obs o{};o.size=1;auto projected=evaluate(s.calibration.data(),p,s.points[0],o,false).residual;
 o.u=float(-projected[0]+2.1);o.v=float(-projected[1]-1.3);
 Data d;d.observations={o};std::array<float,1>sig{1.4f};double e=sfm::pixel_error(s.calibration,p,s.points[0],o)/sig[0];
 if(std::abs(objective(s,d,sig)-rho(e))>1e-10)throw std::runtime_error("whitened objective mismatch");
 o.size=float(sig[0]*std::sqrt(std::max(1.0,e/2)));auto a=evaluate(s.calibration.data(),p,s.points[0],o,true,{});
 // IRLS surrogate gradient must equal the normalized robust objective gradient.
 auto plus=s,minus=s;double h=1e-6;plus.points[0][0]+=h;minus.points[0][0]-=h;
 double numeric=(objective(plus,d,sig)-objective(minus,d,sig))/(2*h);
 auto j=a.jacobian[a.jacobian.size()-3];double analytic=-(a.residual[0]*j[0]+a.residual[1]*j[1])/point_scale(s.points[0]);
 if(std::abs(numeric-analytic)>1e-3*std::max(1.0,std::abs(numeric)))throw std::runtime_error("weighted Huber gradient mismatch");
 std::array<float,1>unit{1};if(std::abs(objective(s,d,unit)-sfm::huber_objective(s,d))>1e-10)throw std::runtime_error("unit weights changed objective");
 std::cout<<"PASS image sigma normalized Huber objective, numeric gradient, unit-weight equivalence\n";
}
}

int wmain(int argc,wchar_t**argv){try{
 if(argc==2&&std::wstring(argv[1])==L"--self-test"){image_ba::self_test();return 0;}
 if(argc!=6||fs::exists(argv[2]))throw std::runtime_error("usage: refine_image_observations INPUT_SFM NEW_OUTPUT REFINED_OBSERVATIONS SIGMA_BIN uniform|weighted");
 fs::path input=argv[1],out=argv[2];std::wstring mode=argv[5];
 if(mode!=L"uniform"&&mode!=L"weighted")throw std::runtime_error("unknown weighting mode");
 auto r=calibrated::load_model(input);auto calibration=r.s.calibration;
 auto original=read_array<sfm::Obs>(input/"observations.bin"),revised=read_array<sfm::Obs>(argv[3]);auto sig=read_array<float>(argv[4]);
 if(original.size()!=revised.size()||sig.size()!=original.size())throw std::runtime_error("measurement count changed");
 std::map<image_ba::Identity,float>sigmas;std::map<image_ba::Identity,sfm::Obs>observations;
 for(unsigned i=0;i<original.size();++i){auto a=original[i],b=revised[i];
  if(a.camera!=b.camera||a.track!=b.track||a.auxiliary!=b.auxiliary)throw std::runtime_error("observation identity changed");
  if(!std::isfinite(b.u)||!std::isfinite(b.v)||std::hypot(double(b.u)-a.u,double(b.v)-a.v)>IMAGE_SHIFT_LIMIT+.001)throw std::runtime_error("image displacement exceeds bound");
  if(!std::isfinite(sig[i])||sig[i]<.5||sig[i]>2)throw std::runtime_error("sigma outside regularized bounds");
  auto key=image_ba::Identity{r.ids.at(a.camera),a.track};b.camera=key.first;b.size=mode==L"weighted"?sig[i]:1;
  if(!observations.emplace(key,b).second)throw std::runtime_error("duplicate point-image observation");sigmas[key]=b.size;
 }
 for(auto&[id,t]:r.tracks)for(auto&o:t.obs)o=observations.at({o.camera,id});
 fs::create_directories(out);std::ofstream log(out/"optimization.jsonl");log<<std::setprecision(17);
 for(unsigned round=0;round<3;++round){log<<"{\"ba_call\":true,\"round\":"<<round<<",\"observation_deletion\":false}\n";image_ba::refine(r,sigmas,log);}
 if(r.s.calibration!=calibration)throw std::runtime_error("fixed intrinsics changed");
 for(auto&[id,t]:r.tracks)if(t.valid)for(auto o:t.obs){auto&p=r.s.poses[r.map.at(o.camera)];if(!(mv(p.rotation,sub(t.xyz,center(p)))[2]>1e-10))throw std::runtime_error("nonpositive depth");}
 graph6::save_model(r,input,out);
 auto saved=read_array<sfm::Obs>(out/"observations.bin");if(saved.size()!=original.size())throw std::runtime_error("observation count changed after BA");
 std::ofstream(out/"image_refinement_audit.json")<<"{\"passed\":true,\"fixed_intrinsics\":true,\"no_observations_deleted\":true,\"no_tracks_reassigned\":true,\"max_coordinate_shift_px\":"<<IMAGE_SHIFT_LIMIT<<",\"weighted\":"<<(mode==L"weighted"?"true":"false")<<",\"observations\":"<<saved.size()<<"}";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
