#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

namespace battery::job {
std::string read_bytes(const std::filesystem::path&, std::uintmax_t maximum);
nlohmann::json read_json(const std::filesystem::path&, std::uintmax_t maximum = 64 * 1024 * 1024);
void atomic_write(const std::filesystem::path&, const std::string&);
void atomic_json(const std::filesystem::path&, const nlohmann::json&);
std::string checksum(const std::string&); // integrity only, not authentication
} // namespace battery::job
