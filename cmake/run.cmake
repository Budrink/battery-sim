add_library(battery_run STATIC
    native/run/src/scenario.cpp
    native/run/src/scenario/material_laws.cpp
    native/run/src/scenario/cells.cpp
    native/run/src/scenario/electrical_circuit.cpp
    native/run/src/scenario/shells.cpp
    native/run/src/scenario/thermal_links.cpp
    native/run/src/scenario/gas_network.cpp
    native/run/src/run_configuration.cpp
    native/run/src/contracts/experiment_contract.cpp
    native/run/src/contracts/contract_validation.cpp
    native/run/src/contracts/gas_combustion_model.cpp
    native/run/src/contracts/contract_assembly.cpp
    native/run/src/contracts/contract_v2.cpp
    native/run/src/contracts/local_ports.cpp
    native/run/src/contracts/electrical_circuit.cpp
    native/run/src/contracts/solved_circuit.cpp
    native/run/src/contracts/experiment_editing.cpp
    native/run/src/contracts/heater_mount.cpp
    native/run/src/contracts/element_editing.cpp
    native/run/src/contracts/geometry_editing.cpp
    native/run/src/contracts/material_editing.cpp
    native/run/src/contracts/shell_editing.cpp
    native/run/src/contracts/vent_patches.cpp
    native/run/src/contracts/enclosure_gas_mesh.cpp
    native/run/src/contracts/enclosure_openings.cpp
    native/run/src/contracts/explicit_enclosure.cpp
    native/run/src/contracts/imported_enclosures.cpp
    native/run/src/contracts/enclosure_editing.cpp
    native/run/src/contracts/editor_document.cpp
)
target_include_directories(battery_run PUBLIC native/run/include)
add_library(battery::run ALIAS battery_run)
target_link_libraries(battery_run PUBLIC battery_native_core nlohmann_json::nlohmann_json)
