#pragma once
#include <cstdint>
#include <vector>

struct Image
{
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;
    bool valid() const { return w > 0 && h > 0 && rgba.size() == size_t(w) * size_t(h) * 4; }
    void clear() { w = h = 0; rgba.clear(); }
};
