#pragma once
namespace battery::ui {
enum class EditorSection {
    Assembly,
    Circuit,
    Contacts,
    Heating,
    Models,
    GasModel,
    Shell,
    Case,
    Run,
    Preview,
    Reference
};
struct EditorSectionInfo {
    const char* title;
    const char* description;
};
inline constexpr EditorSectionInfo editor_sections[] = {
    {"Assembly", "Select a battery, pad or heater. Change its position, size and initial state. Changes here "
                 "affect only that instance; material models may be shared."},
    {"Electrical circuit", "Connect battery terminals and power supplies. The wiring wizard REPLACES the "
                           "network only after confirmation. Existing branch fields update automatically."},
    {"Thermal / gas links",
     "Configure heat paths, contact surfaces and gas mixing. Conductivity, contact conductance and mass "
     "transport are different quantities. Combustion is configured separately in Gas model / R3."},
    {"Heaters / cooling",
     "Place external heaters, set their individual controllers and cutoff, and configure coolant boundaries. "
     "The ramp is a target; actual heating depends on available power and losses."},
    {"Battery materials",
     "Shared battery properties and R1/R2 chemistry. Check which batteries use the model before editing. "
     "Thermal capacity, electrical capacity and chemical energy are different properties."},
    {"Gas model / R3", "One global combustion law for all gas volumes, including generated volumes. Fuel and "
                       "oxygen remain local inventories; this section does not create extra gas or oxygen."},
    {"Shell / valves", "Edit ONE face of the selected battery: plate material, irreversible damage, pressure "
                       "rupture, valve or seam and gas destination. No artificial timed ignition."},
    {"Case / openings",
     "Set case wall properties, cavity gas and exterior openings. Imported explicit meshes retain their "
     "topology; unavailable controls explain why. Add a hole directly or place it on a wall in preview."},
    {"Run / measurements",
     "Set duration, numerical resolution and experimental sensor mapping for the next run. Grid resolution "
     "affects cost and convergence; sensor mapping uncertainty is not a physical fit parameter."},
    {"3D preview",
     "Initial-state preview of the EDITOR document, not the running simulation. Drag to orbit, click to "
     "select, Alt+click for the case. Opening placement uses the actual wall surface."},
    {"Parameter reference",
     "Search the same parameter explanations used by field tooltips. Labels show display units; technical "
     "keys in this reference identify stored contract fields."}};
} // namespace battery::ui
