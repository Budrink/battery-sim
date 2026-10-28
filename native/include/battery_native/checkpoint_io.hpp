#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Local same-build restart format, NOT a portable interchange or scenario file.
// Typed lengths, ABI sizes and a payload checksum detect truncation/corruption.
namespace battery::checkpoint {
inline std::uint64_t checksum(const char* data,std::size_t size) {
    std::uint64_t h=14695981039346656037ull;
    for(std::size_t i=0;i<size;++i){h^=static_cast<unsigned char>(data[i]);h*=1099511628211ull;}
    return h;
}
class Writer {
    std::string data;
public:
    template<class T> void pod(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        data.append(reinterpret_cast<const char*>(&v),sizeof(T));
    }
    void text(const std::string& v){pod(std::uint64_t(v.size()));data.append(v);}
    template<class T> void vector(const std::vector<T>& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        pod(std::uint64_t(sizeof(T)));pod(std::uint64_t(v.size()));
        if(!v.empty())data.append(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
    }
    template<class T> void rows(const std::vector<std::vector<T>>& rows) {
        pod(std::uint64_t(rows.size()));for(const auto& row:rows)vector(row);
    }
    std::string finish(){const auto hash=checksum(data.data(),data.size());pod(hash);return std::move(data);}
};
class Reader {
    const std::string& data;std::size_t position=0,end;
    void need(std::size_t n)const{if(n>end-position)throw std::runtime_error("Truncated restart checkpoint");}
public:
    explicit Reader(const std::string& v):data(v),end(v.size()) {
        if(end<sizeof(std::uint64_t))throw std::runtime_error("Truncated restart checkpoint");
        end-=sizeof(std::uint64_t);std::uint64_t stored;
        std::memcpy(&stored,data.data()+end,sizeof(stored));
        if(stored!=checksum(data.data(),end))throw std::runtime_error("Restart checksum mismatch");
    }
    template<class T> T pod() {
        static_assert(std::is_trivially_copyable_v<T>);need(sizeof(T));T v;
        std::memcpy(&v,data.data()+position,sizeof(T));position+=sizeof(T);return v;
    }
    std::string text(){const auto n=pod<std::uint64_t>();need(n);auto v=data.substr(position,n);position+=n;return v;}
    template<class T> std::vector<T> vector(std::size_t expected) {
        const auto size=pod<std::uint64_t>(),n=pod<std::uint64_t>();
        if(size!=sizeof(T) || n!=expected || n>(end-position)/sizeof(T))
            throw std::runtime_error("Restart topology/count/ABI mismatch");
        std::vector<T> v(n);if(n)std::memcpy(v.data(),data.data()+position,n*sizeof(T));position+=n*sizeof(T);return v;
    }
    template<class T> std::vector<T> vector() {
        const auto saved=position;pod<std::uint64_t>();const auto n=pod<std::uint64_t>();position=saved;
        return vector<T>(n);
    }
    template<class T> std::vector<std::vector<T>> rows(std::size_t expected) {
        if(pod<std::uint64_t>()!=expected)throw std::runtime_error("Restart history topology mismatch");
        std::vector<std::vector<T>> v;for(std::size_t i=0;i<expected;++i)v.push_back(vector<T>());return v;
    }
    void finish()const{if(position!=end)throw std::runtime_error("Trailing restart checkpoint data");}
};
}
