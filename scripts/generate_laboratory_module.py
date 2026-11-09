#!/usr/bin/env python3
"""Emit a synthetic 4x4 fixture using an explicitly sourced fitted chemistry seed."""
import copy
import argparse
import json
import math
from pathlib import Path


def make_module():
    root = Path(__file__).resolve().parents[1]
    e = json.loads((root / "examples/contracts/laboratory-demo.json").read_text())
    e["id"] = "laboratory-module-4x4"
    e["name"] = "4x4 runaway demo / FSRI-derived chemistry / insulating separators"
    e["provenance"]["notes"] = "Synthetic fixture with FSRI-derived material seed. Not experimental validation."
    model = e["battery_models"][0]
    model["geometry"]["dimensions_m"] = [.02, .02, .04]
    preset = apply_material_preset(model, root)
    a = e["assembly"]
    seed_b = copy.deepcopy(a["batteries"][0])
    seed_b.update(model_id=model["id"], model_revision=model["revision"])
    seed_b["initial_state"]["charged_fraction"] = 1
    seed_g = copy.deepcopy(a["gas_network"]["compartments"][0])
    outside = copy.deepcopy(a["gas_network"]["compartments"][1])
    seed_g.update(preset["r3"])
    outside.update(preset["r3"])
    a["gas_network"].update(preset["gas_material"])
    transport = json.loads((root / "examples/contracts/fsri30-gas-transport-preset.json").read_text())
    a["gas_network"].update(copy.deepcopy(transport["network"]))
    # Use the existing stable integration path, not the starter's legacy pairwise default.
    # Mixing routes/areas are generated for THIS fixture, never copied from FSRI.
    a["gas_network"].update(network_integrator="averaged", mixing_integrator="implicit_surfaces",
        reactive_transit=True, averaged_pressure_iterations=8192, averaged_payload_iterations=64,
        coupled_transport_interval_s=.025, free_gas_response_only=True)
    a["batteries"], a["passive_cells"], a["ports"], a["interfaces"], a["vent_connections"] = [], [], [], [], []
    a["gas_network"]["compartments"] = [outside]
    mesh = copy.deepcopy(e["numerics"]["batteries"]["B"])
    e["numerics"]["batteries"] = {}
    pitch, gap, outer_w, outer_h = .024, .002, .022, .042
    positions = {}
    for row in range(4):
        for col in range(4):
            bid = f"B{row * 4 + col:02d}"
            p = [(col - 1.5) * pitch, (row - 1.5) * pitch, 0]
            positions[row, col] = (bid, p)
            b = copy.deepcopy(seed_b)
            b.update(id=bid, gas_compartment="inside-" + bid)
            b["pose"]["position_m"] = p
            a["batteries"].append(b)
            add_internal_gas(a["gas_network"], bid, model, mesh["divisions"], seed_g)
            a["vent_connections"].append(dict(id="vent-" + bid, battery_id=bid, vent_id="top", to="cavity"))
            a["vent_connections"].append(dict(id="vent-bottom-" + bid, battery_id=bid, vent_id="bottom", to="cavity"))
            e["numerics"]["batteries"][bid] = copy.deepcopy(mesh)
    for (row, col), (bid, p) in positions.items():
        for axis, neighbour in ((0, (row, col + 1)), (1, (row + 1, col))):
            if neighbour not in positions:
                continue
            other, q = positions[neighbour]
            pid = f"ISS-{bid}-{other}"
            dimensions = [outer_w, outer_w, outer_h]
            dimensions[axis] = gap
            center = [(x + y) / 2 for x, y in zip(p, q)]
            # x shell panels own corner strips; y-face pads touch their actual smaller face.
            if axis == 1:
                dimensions[0] = .02
            volume = dimensions[0] * dimensions[1] * dimensions[2]
            a["passive_cells"].append(dict(
                id=pid, role="passive", material_id="demo-insulation",
                geometry=dict(kind="prismatic", dimensions_m=dimensions),
                pose=dict(position_m=center, orientation_xyzw=[0, 0, 0, 1]),
                thermal=dict(heat_capacity_j_per_k=volume * 250 * 1000,
                             initial_temperature_k=298.15, conductivity_w_per_m_k=[.06] * 3,
                             divisions=[1, 1, 2], bulk_connectivity=dict(kind="directed_geometry", background_factor=1, active_factor=0)),
                degradation=dict(onset_k=473.15, rate_at_onset_per_s=.001, temperature_scale_k=40),
                runaway=dict(energy_j=0, duration_s=1)))
            u, v = (axis + 1) % 3, (axis + 2) % 3
            area = dimensions[u] * dimensions[v]
            for sign, target in ((-1, bid), (1, other)):
                normal, point = [0, 0, 0], [0, 0, 0]
                normal[axis], point[axis] = sign, sign * gap / 2
                face = ("+" if sign > 0 else "-") + "xy"[axis]
                shell_face = ("-" if sign > 0 else "+") + "xy"[axis]
                port = pid + face
                a["ports"].append(dict(id=port, cell_id=pid, frame="local", surface=face,
                                       normal=normal, centroid_m=point, area_m2=area,
                                       rectangle_m=[dimensions[u], dimensions[v]]))
                conductance = 100 * area  # explicit illustrative shell-to-pad contact, W/K
                a["interfaces"].append(dict(id=port + "-contact", port_a=port,
                    port_b=target + ":case:" + shell_face + ":outer",
                    law=dict(kind="directed_geometry", background_conductance_w_per_k=.01 * conductance,
                             active_conductance_w_per_k=1.98 * conductance)))
    # Four series cells per row, four rows in parallel; source starts OFF.
    circuit = dict(mode="solved_dc", reference_node="negative", nodes=["positive", "negative"], branches=[])
    for row in range(4):
        nodes = ["positive"] + [f"row{row}-n{k}" for k in range(1, 4)] + ["negative"]
        circuit["nodes"] += nodes[1:-1]
        for col in range(4):
            bid = positions[row, col][0]
            circuit["branches"].append(dict(id="battery-" + bid, kind="battery", cell_id=bid,
                                            **{"from": nodes[col], "to": nodes[col + 1]}, emf_v=3.65))
    circuit["branches"].append(dict(id="charger", kind="current_source", **{"from": "negative", "to": "positive"}))
    a["electrical_circuit"] = circuit
    e["conditions"]["source_drives"] = [dict(source_id="charger", current_a=0)]
    e["measurements"] = []
    e["metadata"] = dict(calibration_status="DEMONSTRATION / FSRI-DERIVED MATERIAL SEED / NOT VALIDATED",
        gas_transport_preset=dict(source=transport["source"], source_sha256=transport["source_sha256"],
                                  preset="examples/contracts/fsri30-gas-transport-preset.json",
                                  scope=transport["scope"]),
        material_preset=dict(source=preset["source"], source_sha256=preset["source_sha256"],
                             preset="examples/contracts/fsri30-material-preset.json",
                             scope=preset["scope"]),
        visual_materials={"demo-insulation": "insulation", "demo-thermal-pad": "thermal_conductor"},
        assumptions=["16 prismatic contents 20x20x40 mm, 1 mm proxy steel casing, pitch 24 mm.",
                     "FSRI30 R1/R2/R3 and material properties reused; mass, heat capacity and volatile inventory scale with contents volume.",
                     "Casing remains 1 mm for this fixture (reference 0.18 mm); shell mass follows its actual geometry, not the reference mass.",
                     "24 separate 2 mm insulation pads, k=0.06 W/(m K), rho=250 kg/m3, cp=1000 J/(kg K).",
                     "Contacts h=100 W/(m2 K); damage law is illustrative, not calibrated.",
                     "4S4P DC network, zero input current. Fresh fully charged initial state; no forced ignition or external abuse current.",
                     "B05 lateral foil heater: FSRI30 saved controller, 30 W limit, 0.1 K/s ramp, opposite-shell sensor; no forced ignition.",
                     "Demo weak top/bottom lids: thermal damage onset 450 K, each has a 600 kPa rupture panel and 1 mm valve at 40% strength. Side plates: thermal damage onset 900 K, no declared pressure opening. These strength assumptions are not calibrated.",
                     "120x120x80 mm case, 2 mm walls, one 20 mm circular top opening. Spatial free gas mesh, max 20 mm; pressure-driven exchange with outside air.",
                     "Plastic case walls store heat and contact gas; wall exterior cooling is disabled. Gas can carry heat through openings. Wall-to-wall seam contacts are not inferred."])
    e["numerics"]["simulation"].update(duration_s=3000, max_dt_s=.1, output_interval_s=5)
    add_heater_and_case(e, positions[1, 1][1])
    a["gas_meshes"]["case"]["battery_surface_connectivity_per_m2"] = copy.deepcopy(
        transport["battery_surface_connectivity_per_m2"])
    return e


def apply_material_preset(model, root):
    """Transfer constitutive coefficients, scaling extensive quantities only.

    New prism/case/ports are demo geometry. No archived trajectories, ignition
    times, gas paths or fixture-specific convection coefficients are imported.
    """
    preset = json.loads((root / "examples/contracts/fsri30-material-preset.json").read_text())
    scale = math.prod(model["geometry"]["dimensions_m"]) / preset["reference_volume_m3"]
    model.update(id="fsri-derived-prismatic-demo", revision="1")
    contents = model["contents"]
    contents["material_id"] = "fsri-derived-electrolyte-proxy"
    contents["thermal"] = copy.deepcopy(preset["thermal"])
    contents["thermal"]["heat_capacity_j_per_k"] *= scale
    contents["conversion"] = copy.deepcopy(preset["conversion"])
    contents["conversion"]["mass_kg"] *= scale
    contents["conversion"]["volatile_phase"]["initial_mass_kg"] *= scale
    model["shell"] = copy.deepcopy(preset["shell"])
    model["shell"]["thickness_m"] = .001  # preserve existing fixture dimensions and contacts
    # Mechanical face identity is separate from contents-grid subdivision.
    # Preserve the fitted thermal contacts and conductivities; only the lids
    # inherit the weak damage law. Side plates remain finite-strength material.
    shell = model["shell"]
    weak = {key: shell[key] for key in ("damage_onset_k", "damage_rate_per_s", "damage_scale_k")}
    shell["damage_onset_k"] = 900.
    for face in ("+z", "-z"):
        plate = shell.setdefault("face_materials", {}).setdefault(face, dict(
            material_id="weak-bottom-lid-PROXY",
            conductivity_w_per_m_k=copy.deepcopy(shell["conductivity_w_per_m_k"])))
        plate.update(weak)
    vent = copy.deepcopy(preset["vent"])
    vent["area_m2"] = .02 * .02
    vent["damage_opening"]["valve"] = dict(center_m=[0, 0], radius_m=.001, pressure_factor=.4)
    vent["damage_opening"]["whole_plate"] = True
    model["vents"][0]["channel"] = vent
    bottom = copy.deepcopy(model["vents"][0])
    bottom.update(id="bottom", port_suffix="case:-z:outer")
    model["vents"].append(bottom)
    model["provenance"]["notes"] = (
        "FSRI30 saved material/chemistry seed adapted by volume to a prism; 1 mm casing and valve geometry are demo assumptions. "
        "Electrical coefficients remain illustrative and inactive (source current zero). Not a calibrated commercial battery.")
    return preset


def add_internal_gas(net, bid, model, divisions, seed):
    """One pore inventory per thermal volume; nearest-neighbour internal paths.

    Required by the existing volatile-phase balance: liquid/solid displacement,
    gas temperature and reaction products must be local, not one cold reservoir.
    """
    size = [d/n for d, n in zip(model["geometry"]["dimensions_m"], divisions)]
    count = math.prod(divisions)
    conversion = model["contents"]["conversion"]
    pore = (math.prod(model["geometry"]["dimensions_m"])
            - conversion["mass_kg"]/2300
            - conversion["volatile_phase"]["initial_mass_kg"]/1200) / count
    assert pore > 0
    def index(x, y, z):
        return x + divisions[0]*(y + divisions[1]*z)
    def gas_id(i):
        return "inside-" + bid + (f"-{i}" if i else "")
    for z in range(divisions[2]):
        for y in range(divisions[1]):
            for x in range(divisions[0]):
                i = index(x, y, z)
                g = copy.deepcopy(seed)
                g.update(id=gas_id(i), contents_id=bid, volume_m3=pore,
                    initial_mass_kg=101325*pore/(287*298.15),
                    local_equilibrium=dict(cell_id=bid, thermal_volume=i,
                        reactant_density_kg_m3=2300, residue_density_kg_m3=2000))
                if i:
                    g.pop("contents_id")  # only the fallback source owns the aggregate inventory
                net["compartments"].append(g)
                for axis in range(3):
                    q = [x, y, z]
                    q[axis] += 1
                    if q[axis] >= divisions[axis]:
                        continue
                    j = index(*q)
                    net["channels"].append(dict(id=f"pore-{bid}-{i}-{j}", kind="internal",
                        **{"from": gas_id(i), "to": gas_id(j)},
                        area_m2=size[(axis+1)%3]*size[(axis+2)%3],
                        mixing_length_m=size[axis], mixing_axis=axis, discharge_coefficient=.7,
                        opening_pressure_pa=0))


def add_heater_and_case(e, position):
    """Explicit synthetic apparatus; same six-wall representation as the native case editor."""
    a = e["assembly"]
    preset = json.loads((Path(__file__).resolve().parents[1] / "examples/contracts/fsri30-heater-preset.json").read_text())
    hid = "heater-B05-minusx"
    thickness = preset["thickness_m"]
    area = .022*.042
    a["passive_cells"].append(dict(id=hid, role="passive", material_id="external-resistive-heater",
        geometry=dict(kind="prismatic", dimensions_m=[thickness,.022,.042]),
        pose=dict(position_m=[position[0]-.011-thickness/2,position[1],0], orientation_xyzw=[0,0,0,1]),
        thermal=dict(heat_capacity_j_per_k=preset["heat_capacity_j_per_k"], initial_temperature_k=298.15,
                     conductivity_w_per_m_k=preset["conductivity_w_per_m_k"], divisions=[1]*3,
                     bulk_connectivity=dict(kind="directed_geometry",background_factor=.01,active_factor=1.98,
                                            destroyed_background_factor=1,destroyed_active_factor=0)),
        electrical=dict(resistance_ohm=1, min_resistance_ohm=1, max_resistance_ohm=1,
                        reference_temperature_k=298.15,max_temperature_step_k=2,terminal_axis="z",current_a=0),
        runaway=dict(energy_j=0,duration_s=1)))
    for sign, suffix in ((1,"-contact"),(-1,"-pad")):
        a["ports"].append(dict(id=hid+suffix, cell_id=hid, frame="local", surface="+x" if sign>0 else "-x",
            centroid_m=[sign*thickness/2,0,0], normal=[sign,0,0], area_m2=area, rectangle_m=[.022,.042]))
    # Fit the foil into the existing separator gap, without overlapping solids
    # or keeping a fictitious direct pad-to-battery contact behind the heater.
    pid="ISS-B04-B05"
    pad=next(c for c in a["passive_cells"] if c["id"]==pid)
    old_width=pad["geometry"]["dimensions_m"][0]
    pad["geometry"]["dimensions_m"][0]-=thickness
    pad["pose"]["position_m"][0]-=thickness/2
    pad["thermal"]["heat_capacity_j_per_k"]*=(old_width-thickness)/old_width
    for port in a["ports"]:
        if port["cell_id"]==pid:
            port["centroid_m"][0]=port["normal"][0]*(old_width-thickness)/2
    next(c for c in a["interfaces"] if c["id"]==pid+"+x-contact")["port_b"]=hid+"-pad"
    def law(g):
        return dict(kind="directed_geometry", background_conductance_w_per_k=.01*g,
                    active_conductance_w_per_k=1.98*g)
    a["interfaces"].append(dict(id=hid+"-coupling", port_a=hid+"-contact",
        port_b="B05:case:-x:outer", law=dict(law(preset["contact_h_w_per_m2_k"]*area),
            destroyed_background_conductance_w_per_k=preset["contact_h_w_per_m2_k"]*area,
            destroyed_active_conductance_w_per_k=0,endpoint_damage_weight=0)))
    e["conditions"]["heating_protocols"]=[dict(preset["protocol"],id=hid+"-control",heater_id=hid,sensor_id="B05:case:+x")]
    e["metadata"]["heater_preset"]=dict(preset,heated_battery="B05",foil_face="-x",sensor_face="+x",
        geometry_adaptation="0.2 mm foil in separator gap; ISS-B04-B05 thickness 1.8 mm, unchanged rho/cp/k; no overlapping solids")
    e["metadata"]["visual_materials"]["external-resistive-heater"]="heater"
    p=dict(inside_dimensions_m=[.12,.12,.08],position_m=[0,0,0],thickness_m=.002,
        material_id="demo-case",density_kg_m3=1200,specific_heat_j_kg_k=1200,
        conductivity_w_per_m_k=[.2]*3,initial_temperature_k=298.15,ambient_temperature_k=298.15,
        gas_contact_h_w_m2_k=10,ambient_h_w_m2_k=0,damage_onset_k=500,
        damage_rate_per_s=0,damage_scale_k=40,gas_cell_size_m=.02,
        solid_contact_h_w_m2_k=10,gas_discharge_coefficient=.7)
    size,t=p["inside_dimensions_m"],p["thickness_m"]
    displaced=16*.022*.022*.042 + sum(math.prod(c["geometry"]["dimensions_m"]) for c in a["passive_cells"])
    free_volume=math.prod(size)-displaced
    cavity=a["gas_network"]["compartments"][0]
    cavity["initial_mass_kg"]*=free_volume/cavity["volume_m3"]
    cavity["volume_m3"]=free_volume
    net=a["gas_network"]
    # Exposed top/bottom and perimeter faces only: no gas contact through insulation pads.
    for row in range(4):
        for col in range(4):
            bid=f"B{4*row+col:02d}"
            faces=[("+z",.0004),("-z",.0004)]
            if col==0: faces.append(("-x",.022*.042))
            if col==3: faces.append(("+x",.022*.042))
            if row==0: faces.append(("-y",.020*.042))
            if row==3: faces.append(("+y",.020*.042))
            for face,area in faces:
                net["thermal_contacts"].append(dict(compartment="cavity",port_id=bid+":case:"+face+":outer",connectivity=law(10*area)))
    for index,face in enumerate(["-x","+x","-y","+y","-z","+z"]):
        axis,sign=index//2,1 if index%2 else -1
        u,v=(axis+1)%3,(axis+2)%3
        dims=[t if d==axis else size[d]+(2*t if d>axis else 0) for d in range(3)]
        center=[0,0,0]; center[axis]=sign*(size[axis]+t)/2
        wid="case-"+("plus" if sign>0 else "minus")+"xyz"[axis]
        a["passive_cells"].append(dict(id=wid,role="passive",material_id=p["material_id"],
            geometry=dict(kind="prismatic",dimensions_m=dims),pose=dict(position_m=center,orientation_xyzw=[0,0,0,1]),
            thermal=dict(heat_capacity_j_per_k=math.prod(dims)*1200*1200,initial_temperature_k=298.15,
                         conductivity_w_per_m_k=[.2]*3,divisions=[1]*3),
            degradation=dict(onset_k=500,rate_at_onset_per_s=0,temperature_scale_k=40),
            runaway=dict(energy_j=0,duration_s=1)))
        for outer in (False,True):
            point,normal=[0,0,0],[0,0,0]
            normal[axis]=sign if outer else -sign; point[axis]=normal[axis]*t/2
            rectangle=[dims[u],dims[v]] if outer else [size[u],size[v]]
            area=math.prod(rectangle);pid=wid+(":outer" if outer else ":inner")
            a["ports"].append(dict(id=pid,cell_id=wid,frame="local",surface=("+" if normal[axis]>0 else "-")+"xyz"[axis],
                centroid_m=point,normal=normal,area_m2=area,rectangle_m=rectangle))
            if outer and p["ambient_h_w_m2_k"] > 0:
                e["conditions"]["boundaries"].append(dict(id=pid+"-ambient",port_id=pid,temperature_k=298.15,
                    conductance_w_per_k=p["ambient_h_w_m2_k"]*area,connectivity=law(p["ambient_h_w_m2_k"]*area)))
            elif not outer:
                net["thermal_contacts"].append(dict(compartment="cavity",port_id=pid,connectivity=law(10*area)))
    e["metadata"]["visual_materials"]["demo-case"]="enclosure"
    e["metadata"]["editor_enclosures"]={"case":dict(properties=p,gas_compartment="cavity",free_volume_m3=free_volume)}
    a["gas_meshes"]={"case":dict(compartment="cavity",inside_dimensions_m=size,position_m=[0,0,0],
        maximum_cell_size_m=.02,surface_h_w_m2_k=10,wall_h_w_m2_k=10,discharge_coefficient=.7,
        openings=[dict(id="top-opening",face="+z",shape="circle",center_m=[0.,0.],size_m=[.02,.02])])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Regenerate the checked-in contract")
    args = parser.parse_args()
    payload = json.dumps(make_module(), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
