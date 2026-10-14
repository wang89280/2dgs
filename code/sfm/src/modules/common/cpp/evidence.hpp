#pragma once
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "filters.hpp"

namespace ms {
namespace fs=std::filesystem;
inline std::string json_string(const std::string& s) {
    std::ostringstream o; o<<'"';
    for(unsigned char c:s) {
        if(c=='"' || c=='\\') o<<'\\'<<c;
        else if(c<32) o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else o<<c;
    }
    return o.str()+'"';
}
inline std::vector<std::byte> read_bytes(const fs::path& p) {
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("cannot read "+p.string());
    const auto size=f.tellg(); if(size<0) throw std::runtime_error("invalid file size");
    std::vector<std::byte> b(static_cast<std::size_t>(size));
    f.seekg(0); f.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(b.size()));
    if(!f) throw std::runtime_error("short read "+p.string());
    return b;
}
template<class T> inline std::vector<T> read_array(const fs::path& p) {
    auto b=read_bytes(p); if(b.size()%sizeof(T)) throw std::runtime_error("invalid array size");
    std::vector<T> a(b.size()/sizeof(T)); std::memcpy(a.data(),b.data(),b.size()); return a;
}
inline std::string sha256(std::span<const std::byte> data) {
    BCRYPT_ALG_HANDLE alg{}; BCRYPT_HASH_HANDLE hash{};
    auto ok=[](NTSTATUS s){if(s<0) throw std::runtime_error("BCrypt SHA256 error");};
    ok(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0));
    try {
        ok(BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0));
        for(std::size_t at=0;at<data.size();) {
            const auto size=static_cast<ULONG>((std::min)(data.size()-at,std::size_t(1<<26)));
            ok(BCryptHashData(hash,reinterpret_cast<PUCHAR>(const_cast<std::byte*>(data.data()+at)),size,0)); at+=size;
        }
        std::array<unsigned char,32> digest{}; ok(BCryptFinishHash(hash,digest.data(),32,0));
        BCryptDestroyHash(hash); hash=nullptr; BCryptCloseAlgorithmProvider(alg,0); alg=nullptr;
        std::ostringstream o; for(auto v:digest) o<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(v); return o.str();
    } catch(...) { if(hash) BCryptDestroyHash(hash); if(alg) BCryptCloseAlgorithmProvider(alg,0); throw; }
}
inline void write_bytes(const fs::path& p,std::span<const std::byte> b) {
    std::ofstream f(p,std::ios::binary|std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()),static_cast<std::streamsize>(b.size())); f.close();
    if(!f) throw std::runtime_error("failed writing "+p.string());
}
template<class T> inline void write_array(const fs::path& p,const std::vector<T>& a) { write_bytes(p,std::as_bytes(std::span(a))); }
inline bool compare(std::string_view label,std::span<const std::byte> a,std::span<const std::byte> b) {
    if(a.size()!=b.size()) {std::cerr<<label<<": sizes differ "<<a.size()<<" / "<<b.size()<<'\n'; return false;}
    std::size_t count=0,first=0;
    for(std::size_t i=0;i<a.size();++i) if(a[i]!=b[i]) { if(!count) first=i; ++count; }
    if(count) std::cerr<<label<<": first differing byte="<<first<<", differing bytes="<<count<<'\n';
    return count==0;
}
// Fixed little-endian disk format. Header is independent of compiler packing.
struct Fixture {
    std::uint32_t stage{},scale{1};
    Camera camera{};
    float minimum{},maximum{1};
    Range range{};
    Counts initial{},expected{};
};
inline void write_fixture(const fs::path& dir,const Fixture& x) {
    std::array<std::byte,176> b{};
    auto put=[&](std::size_t at,const auto& v){std::memcpy(b.data()+at,&v,sizeof(v));};
    std::memcpy(b.data(),"MSBOUND1",8); put(8,x.stage); put(12,x.scale);
    std::memcpy(b.data()+16,x.camera.bytes.data(),112);
    put(128,x.minimum); put(132,x.maximum);
    std::uint64_t offset=x.range.offset,count=x.range.count; put(136,offset); put(144,count);
    put(152,x.initial.first); put(156,x.initial.second); put(160,x.expected.first); put(164,x.expected.second);
    write_bytes(dir/"boundary.bin",b);
}
inline Fixture read_fixture(const fs::path& dir) {
    auto b=read_bytes(dir/"boundary.bin");
    if(b.size()!=176 || std::memcmp(b.data(),"MSBOUND1",8)) throw std::runtime_error("invalid boundary header");
    Fixture x;
    auto get=[&](std::size_t at,auto& v){std::memcpy(&v,b.data()+at,sizeof(v));};
    get(8,x.stage); get(12,x.scale); std::memcpy(x.camera.bytes.data(),b.data()+16,112);
    get(128,x.minimum);get(132,x.maximum); get(136,x.range.offset); get(144,x.range.count);
    get(152,x.initial.first);get(156,x.initial.second);get(160,x.expected.first);get(164,x.expected.second);
    return x;
}
inline void publish_manifest(const fs::path& dir,const std::string& provenance,const std::string& extra="") {
    std::ostringstream o; o<<"{\n  \"schema\":1,\n  \"provenance\":"<<json_string(provenance)
        <<",\n  \"pid\":"<<GetCurrentProcessId()<<",\n  \"tid\":"<<GetCurrentThreadId();
    FILETIME ft; GetSystemTimeAsFileTime(&ft); ULARGE_INTEGER t; t.LowPart=ft.dwLowDateTime;t.HighPart=ft.dwHighDateTime;
    o<<",\n  \"timestamp_windows_100ns\":"<<t.QuadPart<<extra<<",\n  \"files\":[";
    bool first=true;
    for(const auto& e:fs::directory_iterator(dir)) {
        if(!e.is_regular_file() || e.path().extension()!=".bin") continue;
        auto b=read_bytes(e.path()); if(!first) o<<','; first=false;
        o<<"\n    {\"name\":"<<json_string(e.path().filename().string())<<",\"bytes\":"<<b.size()<<",\"sha256\":"<<json_string(sha256(b))<<'}';
    }
    o<<"\n  ]\n}\n";
    const auto s=o.str(); write_bytes(dir/"manifest.tmp",std::as_bytes(std::span(s.data(),s.size())));
    fs::rename(dir/"manifest.tmp",dir/"manifest.json");
}
inline bool replay(const fs::path& dir,bool output=true) {
    const auto x=read_fixture(dir); auto d=read_array<float>(dir/(x.stage==4?"normals_before.bin":"depth_before.bin")); auto c=x.initial;
    std::vector<std::uint8_t> mask;
    if(x.stage==1) { const auto cost=read_array<float>(dir/"cost.bin"); check_cost(d,cost,x.range,c); }
    else if(x.stage==2) { mask=read_array<std::uint8_t>(dir/"mask_before.bin"); check_neighbours(d,mask,x.camera,x.scale,x.minimum,x.maximum,x.range,c); }
    else if(x.stage==3) { mask=read_array<std::uint8_t>(dir/"mask_before.bin"); clear_filtered(d,mask,x.range,c); }
    else if(x.stage==4) rotate_normals(d,x.camera,x.range);
    else throw std::runtime_error("unknown stage");
    const auto expected=read_bytes(dir/(x.stage==2?"mask_after.bin":x.stage==4?"normals_after.bin":"depth_after.bin"));
    const auto actual=x.stage==2?std::as_bytes(std::span(mask)):std::as_bytes(std::span(d));
    bool pass=compare("output",actual,expected);
    if(c.first!=x.expected.first || c.second!=x.expected.second) {
        std::cerr<<"counter mismatch: "<<c.first<<','<<c.second<<" expected "<<x.expected.first<<','<<x.expected.second<<'\n'; pass=false;
    }
    if(output) {write_bytes(dir/"replay_output.bin",actual); std::cout<<(pass?"PASS ":"FAIL ")<<dir.string()<<'\n';}
    return pass;
}
}
