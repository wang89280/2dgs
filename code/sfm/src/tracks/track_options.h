#pragma once

// The ablation uses the same geometry and optimizer implementation. Only the
// graph-based recovery operations are disabled; ordinary BA/filtering remain.
struct TrackOptions {
 bool recovery=true;
 unsigned ba_rounds=3;
 static TrackOptions parse(int argc,wchar_t**argv,bool maintenance){
  TrackOptions o;bool replay=false;
  for(int i=4;i<argc;++i){std::wstring a=argv[i];
   if(a==L"--no-track-recovery")o.recovery=false;
   else if(a==L"--track-recovery")o.recovery=true;
   else if(a==L"--ba-rounds"&&maintenance&&i+1<argc){
    std::wstring n=argv[++i];
    if(n.size()!=1||n[0]<L'0'||n[0]>L'3')throw std::runtime_error("BA replay rounds must be 0..3");
    o.ba_rounds=unsigned(n[0]-L'0');replay=true;
   }else throw std::runtime_error("unknown or incomplete track option");
  }
  if(replay&&o.recovery)throw std::runtime_error("--ba-rounds is only valid with --no-track-recovery");
  return o;
 }
};
