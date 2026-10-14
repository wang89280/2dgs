#define GRAPH6_NO_MAIN
#include "refine_graph.cpp"

// Diagnostic and optional final geometric refinement. Original feature pixels
// and track ownership are never changed. Each branch receives three BA calls.
namespace geometry_refine {
double cauchy_objective(const State&s,const Data&d){
 double value=0;for(auto o:d.observations){o.size=1;double e=sfm::pixel_error(s.calibration,s.poses[o.camera],s.points[o.track],o);value+=2*std::log1p(e*e/4);}
 return value;
}
void cauchy(sfm::Reconstruction&r,std::ostream&log){
 sfm::configure_gauge(r);auto d=sfm::collect(r);calibrated::Policy p;p.target=r.s.calibration;p.observation_gate=1e6;
 for(unsigned i=0;i<5;++i){auto previous=r.s;double before=cauchy_objective(r.s,d);
  for(auto&o:d.observations){o.size=1;double e=sfm::pixel_error(r.s.calibration,r.s.poses[o.camera],r.s.points[o.track],o);o.size=float(std::sqrt(1+e*e/4));}
  calibrated::optimize_bounded(r.s,d,p,log);double after=cauchy_objective(r.s,d);bool accept=std::isfinite(after)&&after<=before+1e-8*std::max(1.,before);
  log<<"{\"cauchy_before\":"<<before<<",\"cauchy_after\":"<<after<<",\"accepted\":"<<(accept?"true":"false")<<"}\n";
  if(!accept){r.s=previous;break;}if(before-after<1e-7*std::max(1.,before))break;
 }
 for(auto id:d.active)r.tracks.at(id).xyz=r.s.points[id];
}
void test(){calibrated::self_test();
 double e=3.2,h=1e-6;double numeric=(2*std::log1p((e+h)*(e+h)/4)-2*std::log1p((e-h)*(e-h)/4))/(2*h);
 if(std::abs(numeric-e/(1+e*e/4))>1e-7)throw std::runtime_error("Cauchy influence mismatch");
 std::cout<<"PASS Cauchy influence and calibrated BA derivatives\n";
}
}
int wmain(int argc,wchar_t**argv){try{
 if(argc==2&&std::wstring(argv[1])==L"--self-test"){geometry_refine::test();return 0;}
 if(argc!=4||fs::exists(argv[2]))throw std::runtime_error("usage: refine_geometry INPUT_SFM NEW_OUTPUT fixed|cauchy|calibration");
 fs::path input=argv[1],out=argv[2];std::wstring mode=argv[3];
 if(mode!=L"fixed"&&mode!=L"cauchy"&&mode!=L"calibration")throw std::runtime_error("unknown mode");
 auto r=calibrated::load_model(input);auto original=r.s.calibration;
 fs::create_directories(out);std::ofstream log(out/"optimization.jsonl");log<<std::setprecision(17);
 for(unsigned round=0;round<3;++round){
  log<<"{\"round\":"<<round<<",\"ba_call\":true}\n";
  if(mode==L"cauchy")geometry_refine::cauchy(r,log);
  else {calibrated::Policy p;p.target=original;p.observation_gate=1e6;
   if(mode==L"calibration")p.parameters=round==0?std::vector<unsigned>{0}:(round==1?std::vector<unsigned>{0,6,7,10,11}:std::vector<unsigned>{0,1,2,6,7,10,11});
   calibrated::refine(r,p,log);
  }
 }
 if(mode!=L"calibration"&&r.s.calibration!=original)throw std::runtime_error("fixed intrinsics changed");
 for(auto&[id,t]:r.tracks)if(t.valid)for(auto o:t.obs)if(!(mv(r.s.poses[r.map.at(o.camera)].rotation,sub(t.xyz,center(r.s.poses[r.map.at(o.camera)])))[2]>1e-10))throw std::runtime_error("nonpositive depth");
 graph6::save_model(r,input,out);
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
