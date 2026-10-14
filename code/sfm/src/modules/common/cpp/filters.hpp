#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace ms {
// ABI observed in 2.2.2.21287 PTX, not an asserted full host C++ class layout.
struct alignas(16) Camera {
    std::array<std::byte,112> bytes{};
    std::uint32_t get(std::size_t at) const {
        std::uint32_t v; std::memcpy(&v,bytes.data()+at,4); return v;
    }
    void put(std::size_t at,std::uint32_t v) { std::memcpy(bytes.data()+at,&v,4); }
};
static_assert(sizeof(Camera)==112);
struct Range { std::size_t offset, count; };
struct Counts { std::uint32_t first{},second{}; };
void check_cost(std::span<float> depth,std::span<const float> cost,Range range,Counts& counts);
void check_neighbours(std::span<const float> depth,std::span<std::uint8_t> mask,
    const Camera& camera,std::uint32_t downscale,float minimum,float maximum,Range range,Counts& counts);
void clear_filtered(std::span<float> depth,std::span<const std::uint8_t> mask,Range range,Counts& counts);
// Interleaved float3, matrix at camera byte offset 48. The PTX applies a full
// homogeneous transform; despite the kernel name this is not just a 3x3 rotate.
void rotate_normals(std::span<float> xyz,const Camera& camera,Range range);
}
