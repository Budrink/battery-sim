#pragma once
#include <string_view>

namespace battery::ui {
struct ParameterHelp {
    const char* key;
    const char* title;
    const char* explanation;
};
inline constexpr ParameterHelp parameter_catalog[] = {
#define PARAMETER(key, title, explanation) {key, title, explanation},
#include "editor/help/parameters.def"
#undef PARAMETER
};
inline const ParameterHelp* parameter_help(std::string_view key) {
    for (const auto& field : parameter_catalog)
        if (key == field.key)
            return &field;
    return nullptr;
}
} // namespace battery::ui
