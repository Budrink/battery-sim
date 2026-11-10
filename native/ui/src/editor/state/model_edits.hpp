#pragma once
#include <nlohmann/json.hpp>
#include <map>
#include <string>
#include <optional>

namespace battery::ui {
// Invalid/incomplete form input must not poison the shared experiment document.
// Other panels always edit the last accepted model, not this private form buffer.
class ModelEdits {
  public:
    struct Entry {
        nlohmann::json source, value, attempted;
        std::string error;
    };
    Entry& edit(std::size_t index, const nlohmann::json& model);
    Entry& edit_gas(const nlohmann::json& network);
    void apply_global_gas() {
        gas_apply_ = true;
    }
    void discard_gas() {
        gas_.reset();
        gas_signature_ = nullptr;
        gas_apply_ = false;
    }
    void synchronize(const nlohmann::json& document);
    void commit_valid(nlohmann::json& document);
    bool pending() const;
    void clear() {
        entries_.clear();
        gas_.reset();
        gas_signature_ = nullptr;
        gas_apply_ = false;
    }

  private:
    std::map<std::size_t, Entry> entries_;
    std::optional<Entry> gas_;
    nlohmann::json gas_signature_;
    bool gas_apply_ = false;
};
} // namespace battery::ui
