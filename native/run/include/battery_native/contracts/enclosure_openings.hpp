#pragma once
#include <nlohmann/json_fwd.hpp>
#include <array>
#include <string>
#include <vector>

namespace battery {
// Coordinates on a case face use cyclic axes: X -> YZ, Y -> ZX, Z -> XY.
struct EnclosureOpening {
    std::string id, face;
    int axis = 2, sign = 1;
    bool circle = true;
    std::array<double, 2> center{}, size{};
    double area() const;
    // Exact intersection with a face rectangle, in case-centred coordinates.
    double overlap(std::array<double, 2> low, std::array<double, 2> high) const;
};
std::vector<EnclosureOpening> parse_enclosure_openings(const nlohmann::json& mesh);
// Shared editor descriptors for generated and imported explicit gas networks.
nlohmann::json editable_enclosures(const nlohmann::json& experiment);
// Bind ANY six-wall explicit enclosure to the same opening editor. IDs follow
// the shared case-minusx/case-plusx convention; routes and law are explicit.
nlohmann::json register_explicit_enclosure(const nlohmann::json&, const std::string& case_id,
                                          const nlohmann::json& openings,
                                          const nlohmann::json& original_route_ids,
                                          const nlohmann::json& passage_law);
// Compatibility adapter for older imports missing the enclosure descriptor.
nlohmann::json register_imported_enclosures(const nlohmann::json& experiment);
nlohmann::json edit_explicit_enclosure_openings(const nlohmann::json&, const std::string&,
                                               const nlohmann::json&);
nlohmann::json edit_explicit_enclosure_properties(const nlohmann::json&, const std::string&,
                                                 const nlohmann::json&);
nlohmann::json set_enclosure_openings(const nlohmann::json& experiment, const std::string& case_id,
                                      const nlohmann::json& openings);
} // namespace battery
