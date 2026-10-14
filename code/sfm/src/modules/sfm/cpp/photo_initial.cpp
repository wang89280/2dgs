#include "photo_input.hpp"
#include "initial_calibration.hpp"
#include "projection_decompose.hpp"
#include "pair_refine.hpp"
#include "essential_ransac.hpp"
#include "../../matching/cpp/pose_inputs.hpp"
#include <map>
#include <sstream>
#include <tuple>
using namespace ms;
struct Pair {std::int32_t count{};std::uint32_t padding{};std::uint64_t first{},second{};};
static_assert(sizeof(Pair)==24);
// 142E83480 sorts descending (count, first camera, second camera); padding is ignored.
bool pair_less(const Pair&a,const Pair&b){return std::tie(a.count,a.first,a.second)>std::tie(b.count,b.first,b.second);}
std::pair<unsigned,unsigned> jpeg_size(const fs::path&p){
 auto bytes=read_array<unsigned char>(p);if(bytes.size()>=24&&!std::memcmp(bytes.data(),"\x89PNG\r\n\x1a\n",8)){
  auto be=[&](std::size_t i){return (unsigned(bytes[i])<<24)|(unsigned(bytes[i+1])<<16)|(unsigned(bytes[i+2])<<8)|bytes[i+3];};
  if(be(8)!=13||std::memcmp(bytes.data()+12,"IHDR",4))throw std::runtime_error("invalid PNG IHDR");
  for(std::size_t i=8;i+12<=bytes.size();){auto n=be(i);if(n>bytes.size()-i-12)throw std::runtime_error("truncated PNG chunk");if(!std::memcmp(bytes.data()+i+4,"eXIf",4))throw std::runtime_error("EXIF requires metadata-aware initialization");i+=std::size_t(n)+12;}
  return {be(16),be(20)};
 }if(bytes.size()<4||bytes[0]!=255||bytes[1]!=216)throw std::runtime_error("not JPEG");
 unsigned w=0,h=0;std::size_t i=2;
 while(i<bytes.size()){if(bytes[i++]!=255)throw std::runtime_error("invalid JPEG marker");while(i<bytes.size()&&bytes[i]==255)++i;if(i>=bytes.size())break;auto m=bytes[i++];if(m==218||m==217)break;if(m==1||(m>=208&&m<=215))continue;if(i+2>bytes.size())throw std::runtime_error("truncated JPEG");unsigned n=(unsigned(bytes[i])<<8)|bytes[i+1];if(n<2||i+n>bytes.size())throw std::runtime_error("invalid JPEG segment");
  if(m==225&&n>=8&&!std::memcmp(bytes.data()+i+2,"Exif\0\0",6))throw std::runtime_error("EXIF metadata requires the metadata-aware initialization branch");
  if(m>=192&&m<=207&&m!=196&&m!=200&&m!=204){if(n<8)throw std::runtime_error("short JPEG SOF");h=(unsigned(bytes[i+3])<<8)|bytes[i+4];w=(unsigned(bytes[i+5])<<8)|bytes[i+6];}i+=n;
 }
 if(!w||!h)throw std::runtime_error("JPEG SOF missing");return {w,h};
}
std::vector<Observation> read_observations(const fs::path&p){
 std::ifstream f(p);std::string line;if(!std::getline(f,line)||line!="track,image,feature,u,v,size")throw std::runtime_error("track CSV header");
 std::vector<Observation> out;std::map<std::pair<unsigned,unsigned>,bool> seen;
 while(std::getline(f,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);unsigned track,image,feature;float u,v,size;if(!(s>>track>>image>>feature>>u>>v>>size)||!std::isfinite(u)||!std::isfinite(v)||!(size>0))throw std::runtime_error("invalid observation row");if(!seen.emplace(std::make_pair(image,track),true).second)throw std::runtime_error("duplicate image/track");out.push_back({image,track,UINT64_MAX,u,v,0,size});}
 if(out.empty())throw std::runtime_error("no tracks");std::sort(out.begin(),out.end(),[](auto&a,auto&b){return std::tie(a.image,a.track)<std::tie(b.image,b.track);});return out;
}
int wmain(int argc,wchar_t**argv){try{
 if(argc==4&&std::wstring(argv[1])==L"rank"){auto pairs=read_array<Pair>(argv[2]);std::sort(pairs.begin(),pairs.end(),pair_less);if(fs::exists(argv[3]))throw std::runtime_error("output exists");write_array(argv[3],pairs);return 0;}
 if(argc!=5)throw std::runtime_error("usage: ms_photo_initial IMAGES OWN_OBSERVATIONS_CSV NEW_OUTPUT PAIRS_TO_SOLVE");
 fs::path images=argv[1],out=argv[3];unsigned limit=std::stoul(argv[4]);if(!limit||limit>3160||fs::exists(out))throw std::runtime_error("invalid limit/output");
 auto observations=read_observations(argv[2]);unsigned cameras=observations.back().image+1;if(cameras>80)throw std::runtime_error("dataset entry supports up to 80 numbered photos");
 std::vector<std::array<std::byte,256>> calibrations;std::vector<std::array<double,10>> levels;
 for(unsigned i=0;i<cameras;++i){std::wostringstream name;name<<std::setw(4)<<std::setfill(L'0')<<i+1<<L".jpg";auto [w,h]=jpeg_size(photo_file(images,fs::path(name.str()).stem().string()));
  // Do not silently apply an unverified threshold scaling rule to other formats.
  if(w!=3072||h!=2048)throw std::runtime_error("threshold scaling is capture-verified only for this 3072x2048 dataset");
  auto c=read_array<std::array<std::byte,256>>(fs::path(argv[2]).parent_path()/"calibration_fixed.bin").at(0); if(field<std::uint64_t>(c,8)!=w||field<std::uint64_t>(c,16)!=h)throw std::runtime_error("fixed calibration dimensions mismatch");levels.push_back(photo_initial::thresholds(w,h,field<double>(c,24)));calibrations.push_back(c);
 }
 std::map<unsigned,std::vector<unsigned>> tracks;for(auto&o:observations)tracks[o.track].push_back(o.image);
 std::map<std::pair<unsigned,unsigned>,int> counts;for(auto&[id,views]:tracks)for(unsigned i=0;i<views.size();++i)for(unsigned j=i+1;j<views.size();++j)++counts[{views[i],views[j]}];
 std::vector<Pair>pairs;for(auto&[ids,count]:counts)pairs.push_back({count,0,ids.first,ids.second});std::sort(pairs.begin(),pairs.end(),pair_less);
 fs::create_directories(out);write_array(out/"observations.bin",observations);write_array(out/"calibrations.bin",calibrations);write_array(out/"ranked_pairs.bin",pairs);
 std::ofstream report(out/"pairs.jsonl");report<<std::setprecision(17);unsigned completed=0;
 for(auto pair:pairs){if(completed>=limit)break;if(pair.count<6)continue;auto p=out/("pair_"+std::to_string(pair.first)+"_"+std::to_string(pair.second));fs::create_directories(p);
  std::vector<Pixel>pa,pb;common(observations,pair.first,pair.second,pa,pb);std::vector<Ray>a,b;for(unsigned i=0;i<pa.size();++i){a.push_back(unproject(calibrations[pair.first],pa[i]));b.push_back(unproject(calibrations[pair.second],pb[i]));}
  auto&ta=levels[pair.first];auto&tb=levels[pair.second];write_array(p/"points_first.bin",pa);write_array(p/"points_second.bin",pb);write_array(p/"rays_first.bin",a);write_array(p/"rays_second.bin",b);write_bytes(p/"thresholds_first.bin",std::as_bytes(std::span(ta)));write_bytes(p/"thresholds_second.bin",std::as_bytes(std::span(tb)));write_array(p/"confidence_bits.bin",std::vector<double>{double(0.99f)});
  auto r=essential::estimate(a,b,ta,tb,double(0.99f));auto t=essential::select_level(r.best);auto mask=essential::selected_mask(r,a,b,ta,tb);write_array(p/"best_counts.bin",r.best);write_array(p/"best_matrices.bin",r.matrices);write_array(p/"selected_matrix.bin",std::vector<essential::Matrix>{r.matrices[t]});write_array(p/"selected_mask.bin",mask);
  auto pose=five::relative_pose(r.matrices[t],a,b,mask);write_bytes(p/"first_projection.bin",std::as_bytes(std::span(pose.first)));write_bytes(p/"second_projection.bin",std::as_bytes(std::span(pose.second)));
  auto camera=photo_initial::decompose(pose.second);write_array(p/"rotation.bin",std::vector{camera.rotation});write_array(p/"center.bin",std::vector{camera.center});write_array(p/"projection_intrinsic.bin",std::vector{camera.intrinsic});
  std::vector<std::array<double,4>>points;std::vector<std::uint64_t>ids;for(unsigned i=0;i<a.size();++i)if(mask[i]){points.push_back(five::triangulate(pose.first.data(),pose.second.data(),a[i].data(),b[i].data()));ids.push_back(i);}write_array(p/"triangulated_homogeneous.bin",points);write_array(p/"ray_indices.bin",ids);
  auto refined_mask=mask;auto ka=photo_initial::Matrix{field<double>(calibrations[0],24)+field<double>(calibrations[0],48),0,1536+field<double>(calibrations[0],32),0,field<double>(calibrations[0],24),1024+field<double>(calibrations[0],40),0,0,1},kb=ka;auto refined=photo_initial::refine_relative(camera,a,b,refined_mask,ka,kb,2560.0*double(.004f),2560.0*double(.004f));write_array(p/"refined_mask.bin",refined_mask);write_array(p/"refined_first.bin",std::vector{refined.first});write_array(p/"refined_second.bin",std::vector{refined.second});write_array(p/"refined_rotation.bin",std::vector{refined.camera.rotation});write_array(p/"refined_center.bin",std::vector{refined.camera.center});write_array(p/"refined_intrinsic.bin",std::vector{refined.camera.intrinsic});
  std::vector<unsigned>pair_tracks,refined_tracks;for(auto&o:observations)if(o.image==pair.first&&std::binary_search(tracks.at(o.track).begin(),tracks.at(o.track).end(),unsigned(pair.second)))pair_tracks.push_back(o.track);if(pair_tracks.size()!=a.size())throw std::runtime_error("pair track index extent");std::vector<std::array<double,4>>refined_points;std::vector<std::uint64_t>refined_indices;for(unsigned i=0;i<a.size();++i)if(refined_mask[i]){refined_points.push_back(photo_initial::triangulate_pixels(refined.first,refined.second,refined.pixels_first[i],refined.pixels_second[i]));refined_indices.push_back(i);refined_tracks.push_back(pair_tracks[i]);}write_array(p/"refined_triangulated_homogeneous.bin",refined_points);write_array(p/"refined_ray_indices.bin",refined_indices);write_array(p/"refined_track_ids.bin",refined_tracks);
  report<<"{\"first\":"<<pair.first<<",\"second\":"<<pair.second<<",\"common_tracks\":"<<pair.count<<",\"selected_level\":"<<t<<",\"score_inliers\":"<<r.best[t]<<",\"mask_inliers\":"<<points.size()<<",\"refined_inliers\":"<<refined_points.size()<<",\"refinement_attempts\":"<<refined.optimization.attempts<<",\"refinement_converged\":"<<(refined.optimization.converged?"true":"false")<<",\"iterations\":"<<r.iterations<<"}\n";report.flush();++completed;std::cout<<pair.first<<','<<pair.second<<" : "<<points.size()<<std::endl;
 }
 std::ofstream(out/"summary.json")<<"{\"cameras\":"<<cameras<<",\"tracks\":"<<tracks.size()<<",\"observations\":"<<observations.size()<<",\"ranked_pairs\":"<<pairs.size()<<",\"relative_pairs_solved\":"<<completed<<",\"original_project_read\":false,\"native_calibration_read\":false,\"captured_rays_read\":false,\"native_code_called\":false,\"stable_initial_pair_selected\":false,\"global_camera_initialization_solved\":false,\"metric_scale_available\":false}";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
