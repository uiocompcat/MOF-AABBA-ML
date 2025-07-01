
import pandas as pd
import numpy as np

def load_mof_features(depth_max, unit, walk):
    # Define the base feature names

    # walk: Full, Conn, Func
    # unit: M, S, MS, ...

    # atom-based features
    LA_I = [f'L-A-{walk}-{unit}-feature_identity-{i}' for i in range(depth_max + 1 )]
    LA_ND = [f'L-A-{walk}-{unit}-feature_degree-{i}' for i in range(depth_max + 1 )]
    LA_Z = [f'L-A-{walk}-{unit}-feature_atomic_number-{i}' for i in range(depth_max + 1 )]
    LA_X = [f'L-A-{walk}-{unit}-feature_electronegativity-{i}' for i in range(depth_max + 1 )]
    LA_CR = [f'L-A-{walk}-{unit}-feature_covalent_radius-{i}' for i in range(depth_max + 1 )]
    LA_pol = [f'L-A-{walk}-{unit}-feature_polarizability-{i}' for i in range(depth_max + 1 )]
    
    atom_based_properties = LA_I + LA_ND + LA_Z + LA_X + LA_CR + LA_pol 
  
    # bond-based features
    LB_I = [f'L-B-{walk}-{unit}-feature_identity-{i}' for i in range(depth_max + 1 )]
    LB_d = [f'L-B-{walk}-{unit}-feature_distance-{i}' for i in range(depth_max + 1 )]
    LB_BO = [f'L-B-{walk}-{unit}-feature_bond_order-{i}' for i in range(depth_max + 1 )]

    bond_based_properties = LB_I + LB_d + LB_BO


    # atom-bond-based features
    LAB_I = [f'L-AB-{walk}-{unit}-feature_identity-{i}' for i in range(depth_max + 1 )]
    LAB_ND = [f'L-AB-{walk}-{unit}-feature_degree-{i}' for i in range(depth_max + 1 )]
    LAB_Z = [f'L-AB-{walk}-{unit}-feature_atomic_number-{i}' for i in range(depth_max + 1 )]
    LAB_X = [f'L-AB-{walk}-{unit}-feature_electronegativity-{i}' for i in range(depth_max + 1 )]
    LAB_CR = [f'L-AB-{walk}-{unit}-feature_covalent_radius-{i}' for i in range(depth_max + 1 )]
    LAB_pol = [f'L-AB-{walk}-{unit}-feature_polarizability-{i}' for i in range(depth_max + 1 )]

    atom_bond_based_properties = LAB_I + LAB_ND + LAB_Z + LAB_X + LAB_CR + LAB_pol

    # SBU-based bond features
    S_I = [f'S-SB-Metal-{unit}-feature_identity-{i}' for i in range(depth_max + 1 )]
    S_d = [f'S-SB-Metal-{unit}-feature_distance-{i}' for i in range(depth_max + 1 )]
    S_BO = [f'S-SB-Metal-{unit}-feature_bond_order-{i}' for i in range(depth_max + 1 )]

    sbu_bond_based_properties = S_I + S_d + S_BO
     
    # SBU-feature atom based features
    SA_I = [f'S-A-Metal-{unit}-feature_identity-{i}' for i in range(depth_max + 1 )]
    SA_ND = [f'S-A-Metal-{unit}-feature_degree-{i}' for i in range(depth_max + 1 )]
    SA_Z = [f'S-A-Metal-{unit}-feature_atomic_number-{i}' for i in range(depth_max + 1 )]
    SA_X = [f'S-A-Metal-{unit}-feature_electronegativity-{i}' for i in range(depth_max + 1 )]
    SA_CR = [f'S-A-Metal-{unit}-feature_covalent_radius-{i}' for i in range(depth_max + 1 )]
    SA_pol = [f'S-A-Metal-{unit}-feature_polarizability-{i}' for i in range(depth_max + 1 )]

    sbu_atom_based_properties = SA_I + SA_ND + SA_Z + SA_X + SA_CR + SA_pol

    # geometrical properties
    CellV = ['CellV']
    Df = ['Df']
    Di = ['Di']
    Dif = ['Dif']
    density = ['density']
    SA_vol = ['total_SA_volumetric']
    SA_grav = ['total_SA_gravimetric']
    POV_vol = ['total_POV_volumetric']
    POV_grav = ['total_POV_gravimetric']

    geometric_features = CellV + Df + Di + Dif + density + SA_vol + SA_grav + POV_vol + POV_grav
    print(geometric_features)
    #atom_bond_based_features = [
    #    "S-SB-Metal-M-feature_identiy-0", 
    #    "S-SB-Metal-M-feature_distance-0", 
    #    "S-SB-Metal-M-feature_bond_order-0",
    #    "S-SB-Metal-S-feature_identiy-0", 
    #    "S-SB-Metal-S-feature_distance-0", 
    #    "S-SB-Metal-S-feature_bond_order-0",
    #    "L-AB-Full-M-feature_identiy-0", 
    #    "L-AB-Full-M-feature_degree-0", 
    #    "L-AB-Full-M-feature_atomic_number-0",
    #    "L-AB-Full-M-feature_electronegativity-0", 
    #    "L-AB-Full-M-feature_covalent_radius-0", 
    #    "L-AB-Full-M-feature_polarizability-0",
    #    "L-AB-Full-S-feature_identiy-0", 
    #    "L-AB-Full-S-feature_degree-0", 
    #    "L-AB-Full-S-feature_atomic_number-0",
    #    "L-AB-Full-S-feature_electronegativity-0", 
    #    "L-AB-Full-S-feature_covalent_radius-0", 
    #    "L-AB-Full-S-feature_polarizability-0",
    #    "L-AB-Conn-M-feature_identiy-0", 
    #    "L-AB-Conn-M-feature_degree-0", 
    #    "L-AB-Conn-M-feature_atomic_number-0",
    #    "L-AB-Conn-M-feature_electronegativity-0",
    #    "L-AB-Conn-M-feature_covalent_radius-0", 
    #    "L-AB-Conn-M-feature_polarizability-0",
    #    "L-AB-Conn-S-feature_identiy-0", 
    #    "L-AB-Conn-S-feature_degree-0", 
    #    "L-AB-Conn-S-feature_atomic_number-0",
    #    "L-AB-Conn-S-feature_electronegativity-0", 
    #    "L-AB-Conn-S-feature_covalent_radius-0", 
    #    "L-AB-Conn-S-feature_polarizability-0",
    #    "L-AB-Func-M-feature_identiy-0", 
    #    "L-AB-Func-M-feature_degree-0", 
    #    "L-AB-Func-M-feature_atomic_number-0",
    #    "L-AB-Func-M-feature_electronegativity-0", 
    #    "L-AB-Func-M-feature_covalent_radius-0", 
    #    "L-AB-Func-M-feature_polarizability-0",
    #    "L-AB-Func-S-feature_identiy-0", 
    #    "L-AB-Func-S-feature_degree-0", 
    #    "L-AB-Func-S-feature_atomic_number-0",
    #    "L-AB-Func-S-feature_electronegativity-0", 
    #    "L-AB-Func-S-feature_covalent_radius-0", 
    #    "L-AB-Func-S-feature_polarizability-0"
    #]
    #
    #atom_based_features = [
#
    #    "S-A-Metal-M-feature_identiy-0", 
    #    "S-A-Metal-M-feature_degree-0", 
    #    "S-A-Metal-M-feature_atomic_number-0",
    #    "S-A-Metal-M-feature_electronegativity-0", 
    #    "S-A-Metal-M-feature_covalent_radius-0", 
    #    "S-A-Metal-M-feature_polarizability-0",
    #    "S-A-Metal-S-feature_identiy-0", 
    #    "S-A-Metal-S-feature_degree-0", 
    #    "S-A-Metal-S-feature_atomic_number-0",
    #    "S-A-Metal-S-feature_electronegativity-0", 
    #    "S-A-Metal-S-feature_covalent_radius-0", 
    #    "S-A-Metal-S-feature_polarizability-0"
    #    "L-A-Full-M-feature_identiy-0", 
    #    "L-A-Full-M-feature_degree-0", 
    #    "L-A-Full-M-feature_atomic_number-0",
    #    "L-A-Full-M-feature_electronegativity-0",
    #    "L-A-Full-M-feature_covalent_radius-0", 
    #    "L-A-Full-M-feature_polarizability-0",
    #    "L-A-Full-S-feature_identiy-0",
    #    "L-A-Full-S-feature_degree-0",
    #    "L-A-Full-S-feature_atomic_number-0",
    #    "L-A-Full-S-feature_electronegativity-0", 
    #    "L-A-Full-S-feature_covalent_radius-0", 
    #    "L-A-Full-S-feature_polarizability-0",
    #    "L-A-Conn-M-feature_identiy-1", 
    #    "L-A-Conn-M-feature_degree-0", 
    #    "L-A-Conn-M-feature_atomic_number-0",
    #    "L-A-Conn-M-feature_electronegativity-0", 
    #    "L-A-Conn-M-feature_covalent_radius-0", 
    #    "L-A-Conn-M-feature_polarizability-0",
    #    "L-A-Conn-S-feature_identiy-0", 
    #    "L-A-Conn-S-feature_degree-0", 
    #    "L-A-Conn-S-feature_atomic_number-0",
    #    "L-A-Conn-S-feature_electronegativity-0", 
    #    "L-A-Conn-S-feature_covalent_radius-0", 
    #    "L-A-Conn-S-feature_polarizability-0",
    #    "L-A-Func-M-feature_identiy-0", 
    #    "L-A-Func-M-feature_degree-0", 
    #    "L-A-Func-M-feature_atomic_number-0",
    #    "L-A-Func-M-feature_electronegativity-0", 
    #    "L-A-Func-M-feature_covalent_radius-0", dentiy-0", 
    #    "S-A-Metal-S-feature_degree-0", 
    #    "S-A-Metal-S-feature_atomic_number-0",
    #    "S-A-Metal-S-feature_electronegativity-0", 
    #    "S-A-Metal-S-feature_covalent_radius-0", 
    #    "S-A-Metal-S-feature_polarizability-0"
    #    "L-A-Func-M-feature_polarizability-0",
    #    "L-A-Func-S-feature_identiy-0", 
    #    "L-A-Func-S-feature_degree-0", 
    #    "L-A-Func-S-feature_atomic_number-0",
    #    "L-A-Func-S-feature_electronegativity-0", 
    #    "L-A-Func-S-feature_covalent_radius-0", 
    #    "L-A-Func-S-feature_polarizability-0",
#
#
    #    ]
#
    #
    #bond_based_features = [
    #    "L-B-Full-M-feature_identiy-0", 
    #    "L-B-Full-M-feature_distance-0", 
    #    "L-B-Full-M-feature_bond_order-0",
#
    #    "L-B-Full-S-feature_identiy-0", 
    #    "L-B-Full-S-feature_distance-0", 
    #    "L-B-Full-S-feature_bond_order-0",
#
    #    "L-B-Conn-M-feature_identiy-0", 
    #    "L-B-Conn-M-feature_distance-0", 
    #    "L-B-Conn-M-feature_bond_order-0",
#
    #    "L-B-Conn-S-feature_identiy-0", 
    #    "L-B-Conn-S-feature_distance-0", 
    #    "L-B-Conn-S-feature_bond_order-0",
#
    #    "L-B-Func-M-feature_identiy-0", 
    #    "L-B-Func-M-feature_distance-0", 
    #    "L-B-Func-M-feature_bond_order-0",
#
    #    "L-B-Func-S-feature_identiy-0", 
    #    "L-B-Func-S-feature_distance-0", 
    #    "L-B-Func-S-feature_bond_order-0"
    #]
#
    all_aabba_features = [atom_based_properties, 
                        bond_based_properties, 
                        atom_bond_based_properties,
                        sbu_atom_based_properties,
                        sbu_bond_based_properties,
                        geometric_features]

    ## Generate the full list of column names by replacing the trailing 0 with 0-10
    #def generate_columns(features):
    #    columns = []
    #    for feature in features:
    #        base = feature[:-1]
    #        columns.extend([f"{base}{i}" for i in range(1)])
    #    return columns

    #all_aabba_features = (
    #    generate_columns(atom_based_features) + 
    #    generate_columns(bond_based_features) + 
    #    generate_columns(atom_bond_based_features)
    #)
    
    return all_aabba_features