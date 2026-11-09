#include "artifacts.hpp"
#include "battery_native/io/checkpoint_io.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace battery::job {
std::string read_bytes(const std::filesystem::path& path, std::uintmax_t maximum) {
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("Not a regular file: " + path.string());
    const auto size = std::filesystem::file_size(path);
    if (size > maximum)
        throw std::runtime_error("Input file exceeds size limit: " + path.string());
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ios::failbit | std::ios::badbit);
    std::string result(static_cast<std::size_t>(size), '\0');
    if (size)
        input.read(result.data(), static_cast<std::streamsize>(size));
    return result;
}
nlohmann::json read_json(const std::filesystem::path& path, std::uintmax_t maximum) {
    return nlohmann::json::parse(read_bytes(path, maximum));
}
void atomic_write(const std::filesystem::path& path, const std::string& bytes) {
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(temporary, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    std::filesystem::rename(temporary, path);
}
void atomic_json(const std::filesystem::path& path, const nlohmann::json& value) {
    atomic_write(path, value.dump(2) + "\n");
}
std::string checksum(const std::string& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << checkpoint::checksum(bytes.data(), bytes.size());
    return out.str();
}
} // namespace battery::job
