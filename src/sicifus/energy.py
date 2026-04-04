import subprocess
import polars as pl
import numpy as np
from pathlib import Path
import shutil
import uuid
import os
from typing import Optional, Tuple, Dict

class XTBScorer:
    """
    Interface for running semi-empirical QM calculations using GFN2-xTB.
    Used to score ligand binding stability within a protein pocket.
    """
    
    def __init__(self, work_dir: str = "./xtb_work", keep_files: bool = False):
        """
        Args:
            work_dir: Directory where xTB calculations will run.
            keep_files: If True, calculation files (xyz, out, etc.) are preserved.
                        If False, they are deleted after scoring.
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.keep_files = keep_files
        self.debug_dir = self.work_dir / "debug_structures"
        if self.keep_files:
            self.debug_dir.mkdir(exist_ok=True, parents=True)
        self._check_xtb_available()

    def _check_xtb_available(self):
        if shutil.which("xtb") is None:
            print("Warning: 'xtb' binary not found in PATH. Energy scoring will fail.")
            print("  Install via conda: conda install -c conda-forge xtb")

    def _add_hydrogens_to_ligand(self, ligand_df: pl.DataFrame, ligand_name: str) -> pl.DataFrame:
        """
        Uses Meeko (preferred), RDKit, or OpenBabel to add hydrogens to the ligand.
        Returns a new DataFrame with hydrogens added.
        """
        # 1. Try Meeko first (Best for state enumeration and protonation)
        meeko_success = False
        new_df = None
        
        # Use the first row of original DF as template for metadata
        template = ligand_df.row(0, named=True)
        
        # --- Helper: build RDKit mol from DataFrame coordinates ---
        def _build_mol_from_df(df):
            """Build an RDKit mol from a DataFrame of heavy atoms, with bond perception.
            Uses the same strategy as build_ligand_mol in analysis.py:
            1. Try rdDetermineBonds (full: connectivity + bond orders)
            2. Try rdDetermineBonds (connectivity only, no Hueckel)
            3. Fallback: distance-based bond assignment (0.8-1.85 Å = bonded)
            """
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            elements = df["element"].to_list()
            coords = df.select(["x", "y", "z"]).to_numpy()
            names = df["atom_name"].to_list()
            
            mol = Chem.RWMol()
            conf = Chem.Conformer(len(elements))
            
            for i, (elem, coord) in enumerate(zip(elements, coords)):
                atom = Chem.Atom(elem)
                idx = mol.AddAtom(atom)
                conf.SetAtomPosition(idx, (float(coord[0]), float(coord[1]), float(coord[2])))
                
            mol.AddConformer(conf)
            
            # Strategy 1: Full bond perception (connectivity + bond orders)
            bonds_ok = False
            try:
                from rdkit.Chem import rdDetermineBonds
                rdDetermineBonds.DetermineConnectivity(mol)
                rdDetermineBonds.DetermineBondOrders(mol)
                bonds_ok = mol.GetNumBonds() > 0
            except Exception as e:
                print(f"  [Ligand H] rdDetermineBonds failed: {e}")
            
            # Strategy 2: Connectivity only (no Hueckel)
            if not bonds_ok:
                try:
                    from rdkit.Chem import rdDetermineBonds
                    # Reset mol — rdDetermineBonds may have left it in a bad state
                    mol2 = Chem.RWMol()
                    conf2 = Chem.Conformer(len(elements))
                    for i, (elem, coord) in enumerate(zip(elements, coords)):
                        atom = Chem.Atom(elem)
                        idx = mol2.AddAtom(atom)
                        conf2.SetAtomPosition(idx, (float(coord[0]), float(coord[1]), float(coord[2])))
                    mol2.AddConformer(conf2)
                    rdDetermineBonds.DetermineConnectivity(mol2, useHueckel=False)
                    if mol2.GetNumBonds() > 0:
                        mol = mol2
                        bonds_ok = True
                except Exception:
                    pass
            
            # Strategy 3: Distance-based fallback (same as build_ligand_mol in analysis.py)
            # Any two atoms 0.8-1.85 Å apart are assumed bonded (covers most organic bonds)
            if not bonds_ok:
                print(f"  [Ligand H] rdDetermineBonds failed. Using distance-based bond assignment...")
                from scipy.spatial.distance import cdist as cdist_fn
                dists = cdist_fn(coords, coords)
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        if 0.8 < dists[i, j] < 1.85:
                            mol.AddBond(i, j, Chem.BondType.SINGLE)
                bonds_ok = mol.GetNumBonds() > 0
                
                if bonds_ok:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        pass
            
            n_bonds = mol.GetNumBonds()
            print(f"  [Ligand H] Built mol: {mol.GetNumAtoms()} atoms, {n_bonds} bonds")
            
            return mol.GetMol(), names, bonds_ok
        
        # Strip existing hydrogens first (shared across all methods)
        ligand_heavy = ligand_df.filter(pl.col("element") != "H")
        if ligand_heavy.height == 0:
            return ligand_df
        
        n_heavy = ligand_heavy.height
        
        try:
            from meeko import MoleculePreparation
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol, atom_names, bonds_ok = _build_mol_from_df(ligand_heavy)
            
            if not bonds_ok:
                print(f"  [Ligand H] Skipping Meeko (no bonds detected)")
                raise ValueError("No bonds detected, cannot use Meeko")
            
            # Use Meeko to prepare the molecule (adds hydrogens, handles tautomers/states)
            # CRITICAL: merge_these_atom_types=() prevents Meeko from merging non-polar hydrogens
            # into heavy atoms (United Atom model), forcing All Atom model for QM.
            preparator = MoleculePreparation(merge_these_atom_types=())
            setups = preparator.prepare(mol)
            
            if setups:
                mol_h = setups[0].mol
                n_h_new = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
                
                if n_h_new == 0:
                    print(f"  [Ligand H] Meeko returned 0 hydrogens, falling through...")
                    raise ValueError("Meeko added no hydrogens")
                
                # Check if coordinates are valid
                if mol_h.GetNumConformers() > 0:
                    conf_h = mol_h.GetConformer()
                    positions = conf_h.GetPositions()
                    
                    if not (np.all(positions == 0.0) or np.any(np.isnan(positions))):
                        meeko_success = True
                        
                        new_rows = []
                        h_counter = 0
                        heavy_idx = 0
                        for i, atom in enumerate(mol_h.GetAtoms()):
                            pos = conf_h.GetAtomPosition(i)
                            elem = atom.GetSymbol()
                            # Preserve original names for heavy atoms, generate H1, H2... for new hydrogens
                            if atom.GetAtomicNum() > 1 and heavy_idx < len(atom_names):
                                name = atom_names[heavy_idx]
                                heavy_idx += 1
                            else:
                                h_counter += 1
                                name = f"H{h_counter}"
                            
                            new_rows.append({
                                "structure_id": template["structure_id"],
                                "model": template["model"],
                                "chain": template["chain"],
                                "residue_name": template["residue_name"],
                                "residue_number": template["residue_number"],
                                "atom_name": name,
                                "x": pos.x,
                                "y": pos.y,
                                "z": pos.z,
                                "b_factor": template["b_factor"],
                                "element": elem
                            })
                        new_df = pl.DataFrame(new_rows)
                    else:
                        print(f"  [Ligand H] Meeko produced invalid coords")
                        raise ValueError("Invalid coordinates from Meeko")

        except ImportError:
            print("  [Ligand H] Meeko not installed, trying RDKit...")
        except Exception as e:
            print(f"  [Ligand H] Meeko failed: {e}")

        if meeko_success and new_df is not None:
            n_h = new_df.filter(pl.col("element") == "H").height
            print(f"  [Ligand H] Meeko success: added {n_h} H ({n_heavy} → {new_df.height} atoms)")
            return new_df

        # 2. Try RDKit (Fallback)
        rdkit_success = False
        new_df = None
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Rebuild mol if we don't have one (Meeko path may have failed before building)
            try:
                _ = mol.GetNumAtoms()
            except (NameError, AttributeError):
                mol, atom_names, bonds_ok = _build_mol_from_df(ligand_heavy)
            
            # Even without bonds, RDKit AddHs can still work (it uses valence rules)
            # But it needs bonds to know the correct valence
            if mol.GetNumBonds() == 0:
                print(f"  [Ligand H] WARNING: RDKit mol has 0 bonds. AddHs will likely add nothing or wrong H count.")
            
            # Add Hydrogens
            mol_h = Chem.AddHs(mol, addCoords=True)
            n_h_added = mol_h.GetNumAtoms() - mol.GetNumAtoms()
            print(f"  [Ligand H] RDKit AddHs: {n_h_added} hydrogens added")
            
            if n_h_added == 0:
                print(f"  [Ligand H] RDKit added 0 H (likely no bonds). Falling through to OpenBabel...")
                raise ValueError("RDKit added no hydrogens")
            
            # Refine Hydrogen Positions (Fix "Protons in Space")
            try:
                if mol_h.GetNumConformers() > 0:
                    conf = mol_h.GetConformer()
                    coord_map = {}
                    for atom in mol_h.GetAtoms():
                        if atom.GetAtomicNum() > 1: # Heavy atom
                            idx = atom.GetIdx()
                            pos = conf.GetAtomPosition(idx)
                            coord_map[idx] = pos
                    
                    result = AllChem.EmbedMolecule(mol_h, coordMap=coord_map, forceTol=0.01, useRandomCoords=True)
                    
                    if result == -1:
                        print(f"  [Ligand H] RDKit constrained embedding failed (code -1), using AddHs coords")
                    else:
                        print(f"  [Ligand H] RDKit constrained embedding succeeded")
            except Exception as e:
                print(f"  [Ligand H] RDKit constrained embedding error: {e}")
            
            # Check if coordinates are valid (not all 0.0 or NaN)
            conf_h = mol_h.GetConformer()
            positions = conf_h.GetPositions()
            
            if np.all(positions == 0.0) or np.any(np.isnan(positions)):
                rdkit_success = False
                print(f"  [Ligand H] RDKit produced invalid coords (all zeros or NaN)")
            else:
                rdkit_success = True
                
                # Convert back to DataFrame — preserve original heavy atom names
                new_rows = []
                heavy_idx = 0
                h_counter = 0
                for i, atom in enumerate(mol_h.GetAtoms()):
                    pos = conf_h.GetAtomPosition(i)
                    elem = atom.GetSymbol()
                    
                    if atom.GetAtomicNum() > 1 and heavy_idx < len(atom_names):
                        name = atom_names[heavy_idx]
                        heavy_idx += 1
                    else:
                        h_counter += 1
                        name = f"H{h_counter}"
                        
                    new_rows.append({
                        "structure_id": template["structure_id"],
                        "model": template["model"],
                        "chain": template["chain"],
                        "residue_name": template["residue_name"],
                        "residue_number": template["residue_number"],
                        "atom_name": name,
                        "x": pos.x,
                        "y": pos.y,
                        "z": pos.z,
                        "b_factor": template["b_factor"],
                        "element": elem
                    })
                new_df = pl.DataFrame(new_rows)
                
        except ImportError:
            print("  [Ligand H] RDKit not installed, trying OpenBabel...")
        except Exception as e:
            print(f"  [Ligand H] RDKit failed: {e}")

        if rdkit_success and new_df is not None:
            n_h = new_df.filter(pl.col("element") == "H").height
            print(f"  [Ligand H] RDKit success: added {n_h} H ({n_heavy} → {new_df.height} atoms)")
            return new_df

        # 3. Fallback to OpenBabel
        if shutil.which("obabel"):
            print("  [Ligand H] Trying OpenBabel fallback...")
            try:
                # Write temp PDB
                pdb_content = self._df_to_pdb(ligand_df)
                xyz_h_content = self._add_hydrogens_openbabel(pdb_content, input_format="pdb", pH=7.4)
                
                # Parse XYZ back to DataFrame
                lines = xyz_h_content.splitlines()
                try:
                    n_atoms = int(lines[0])
                    new_rows = []
                    # XYZ format: Element X Y Z
                    for line in lines[2:]:
                        parts = line.split()
                        if len(parts) >= 4:
                            elem = parts[0]
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            new_rows.append({
                                "atom_name": elem, # Lost original names in XYZ roundtrip, but acceptable for ligand
                                "residue_name": ligand_name,
                                "element": elem,
                                "x": x, "y": y, "z": z
                            })
                    if len(new_rows) > 0:
                        return pl.DataFrame(new_rows)
                except ValueError:
                    pass
            except Exception as e:
                print(f"  [Ligand H] OpenBabel fallback failed: {e}")
        else:
            print("  [Ligand H] OpenBabel not found in PATH")

        print(f"  [Ligand H] WARNING: All protonation methods failed! Returning unprotonated ligand ({ligand_df.height} atoms, {ligand_df.filter(pl.col('element') == 'H').height} H)")
        return ligand_df

    def _df_to_pdb(self, df: pl.DataFrame) -> str:
        """
        Converts a DataFrame of atoms to a PDB formatted string.
        Ensures strict column alignment for OpenBabel compatibility.
        """
        lines = []
        for i, row in enumerate(df.iter_rows(named=True)):
            atom_name = str(row.get('atom_name', 'X')).strip()
            res_name = str(row.get('residue_name', 'UNK')).strip()[:3]
            chain_id = str(row.get('chain_id', 'A')).strip()[:1] # Truncate to 1 char
            res_seq = row.get('residue_number', 1)
            try:
                res_seq = int(res_seq)
            except:
                res_seq = 1
                
            x, y, z = row['x'], row['y'], row['z']
            elem = str(row.get('element', atom_name[0])).strip().upper()
            
            # Atom Name Alignment Logic (PDB Standard)
            # 4 chars: start col 13 (e.g. 1HG1)
            # <4 chars: start col 14 (e.g. " N  ", " CA ")
            # UNLESS 2-letter element (e.g. FE): start col 13 ("FE  ")
            
            if len(atom_name) >= 4:
                aname_fmt = f"{atom_name[:4]}"
            elif len(elem) == 2: # 2-letter element starts at 13
                aname_fmt = f"{atom_name:<4}"
            else: # 1-letter element starts at 14
                aname_fmt = f" {atom_name:<3}"

            # ATOM      1  N   ALA A  13      14.309  14.283  16.890  1.00 28.66           N
            # 1-6: "ATOM  "
            # 7-11: Serial number (i+1)
            # 13-16: Atom name (aname_fmt)
            # 17: AltLoc (space)
            # 18-20: ResName
            # 22: ChainID
            # 23-26: ResSeq
            # 31-38: X
            # 39-46: Y
            # 47-54: Z
            # 77-78: Element
            
            line = (f"ATOM  {i+1:>5} {aname_fmt:<4} {res_name:<3} {chain_id:>1}{res_seq:>4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2}")
            lines.append(line)
        return "\n".join(lines)

    def _add_hydrogens_openbabel(self, content: str, input_format: str = "pdb", pH: float = 7.4) -> str:
        """
        Uses OpenBabel to add hydrogens to a structure at a specific pH.
        Returns the new content in XYZ format (for xTB compatibility).
        """
        if shutil.which("obabel") is None:
            return content

        run_id = str(uuid.uuid4())[:8]
        run_dir = self.work_dir / f"ob_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        in_file = run_dir / f"input.{input_format}"
        out_file = run_dir / "protonated.xyz"
        
        with open(in_file, "w") as f:
            f.write(content)
            
        # obabel input.pdb -O protonated.xyz -p 7.4
        cmd = ["obabel", str(in_file), "-O", str(out_file), "-p", str(pH)]
        
        try:
            process = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
            if process.returncode != 0:
                print(f"  OpenBabel Error (exit {process.returncode}):")
                print(f"    Command: {' '.join(cmd)}")
                print(f"    Stderr: {process.stderr.strip()}")
                if input_format == "pdb":
                    print("    Input PDB snippet (first 5 lines):")
                    print("\n".join(content.splitlines()[:5]))
            
            if out_file.exists() and out_file.stat().st_size > 0:
                return out_file.read_text()
            else:
                print("  OpenBabel failed: Output file empty or missing.")
                
        except Exception as e:
            print(f"  OpenBabel execution failed: {e}")
        finally:
            if not self.keep_files:
                try:
                    shutil.rmtree(run_dir)
                except: pass
                
        return content

    def extract_pocket_xyz(self, all_atoms_df: pl.DataFrame, ligand_df: pl.DataFrame, 
                           ligand_name: str, distance_cutoff: float = 6.0) -> Tuple[Optional[str], Optional[str], Optional[str], int]:
        """
        Extracts pocket + ligand atoms and returns them as XYZ strings.
        Returns THREE xyz strings: (complex, protein_only, ligand_only).
        Also returns the number of protein atoms (to freeze them later).
        
        Args:
            all_atoms_df: DataFrame of all protein atoms.
            ligand_df: DataFrame of ligand atoms.
            ligand_name: Residue name of the ligand.
            distance_cutoff: Angstrom radius around ligand to include.
            
        Returns:
            (xyz_complex, xyz_protein, xyz_ligand, n_protein_atoms)
        """
        target_ligand = ligand_df.filter(pl.col("residue_name") == ligand_name)
        if target_ligand.height == 0:
            return None, None, None, 0

        # 1. Get coordinates
        lig_coords = target_ligand.select(["x", "y", "z"]).to_numpy()
        prot_coords = all_atoms_df.select(["x", "y", "z"]).to_numpy()
        
        if len(prot_coords) == 0:
            return None, None, None, 0
            
        # 2. Find pocket atoms (geometry search)
        # IMPORTANT: We select COMPLETE residues — if any atom of a residue is within
        # the cutoff, we include ALL atoms of that residue. This prevents dangling bonds
        # and fragmented amino acids that crash xTB.
        from scipy.spatial.distance import cdist
        dists = cdist(prot_coords, lig_coords)
        min_dists = np.min(dists, axis=1)
        
        # First pass: find atoms within cutoff
        close_mask = min_dists < distance_cutoff
        close_atoms = all_atoms_df.filter(close_mask)
        
        if close_atoms.height == 0:
            print(f"Warning: No protein atoms found within {distance_cutoff}A of {ligand_name}")
            return None, None, None, 0
        
        # Second pass: expand to complete residues
        # Get unique (chain, residue_number, residue_name) tuples that have at least one atom in range
        residue_id_cols = ["chain", "residue_number", "residue_name"]
        # Some columns might be missing; use what's available
        available_cols = [c for c in residue_id_cols if c in all_atoms_df.columns]
        
        if available_cols:
            touched_residues = close_atoms.select(available_cols).unique()
            # Join back to get ALL atoms from those residues
            pocket_atoms = all_atoms_df.join(touched_residues, on=available_cols, how="semi")
        else:
            # Fallback: no residue info, just use atom-level selection
            pocket_atoms = close_atoms
        
        if pocket_atoms.height == 0:
            print(f"Warning: No protein atoms found within {distance_cutoff}A of {ligand_name}")
            return None, None, None, 0
        
        n_residues = touched_residues.height if available_cols else 0
        n_atom_level = close_atoms.height
        print(f"  Pocket: {n_atom_level} atoms within {distance_cutoff}A → {n_residues} complete residues → {pocket_atoms.height} total atoms")
        
        # Helper to generate XYZ string
        def _to_xyz(df, comment):
            lines = [f"{df.height}", comment]
            for row in df.iter_rows(named=True):
                elem = row['element']
                if not elem:
                    elem = row['atom_name'][0]
                lines.append(f"{elem:<2} {row['x']:12.6f} {row['y']:12.6f} {row['z']:12.6f}")
            return "\n".join(lines)

        # Check for hydrogens in protein
        prot_has_h = pocket_atoms.filter(pl.col("element") == "H").height > 0
        lig_has_h = target_ligand.filter(pl.col("element") == "H").height > 0
        
        xyz_prot_body = ""
        xyz_lig_body = ""
        n_prot_final = 0
        n_lig_final = 0
        
        # --- PROTEIN PROTONATION ---
        if not prot_has_h:
            if shutil.which("obabel"):
                print("  Protonating protein pocket (OpenBabel pH 7.4)...")
                # Convert to PDB to preserve residue info for correct protonation
                pdb_content = self._df_to_pdb(pocket_atoms)
                xyz_prot_h_str = self._add_hydrogens_openbabel(pdb_content, input_format="pdb")
                
                # Parse the resulting XYZ
                lines = xyz_prot_h_str.splitlines()
                try:
                    n_prot_final = int(lines[0])
                    xyz_prot_body = "\n".join(lines[2:])
                except ValueError:
                    # Fallback if OB failed
                    print("  Warning: OpenBabel output invalid. Using unprotonated protein.")
                    xyz_prot_raw = _to_xyz(pocket_atoms, "")
                    lines = xyz_prot_raw.splitlines()
                    n_prot_final = int(lines[0])
                    xyz_prot_body = "\n".join(lines[2:])
            else:
                print("  Warning: Protein lacks hydrogens and OpenBabel not found.")
                xyz_prot_raw = _to_xyz(pocket_atoms, "")
                lines = xyz_prot_raw.splitlines()
                n_prot_final = int(lines[0])
                xyz_prot_body = "\n".join(lines[2:])
        else:
            xyz_prot_raw = _to_xyz(pocket_atoms, "")
            lines = xyz_prot_raw.splitlines()
            n_prot_final = int(lines[0])
            xyz_prot_body = "\n".join(lines[2:])

        # --- LIGAND PROTONATION ---
        if not lig_has_h:
            print(f"  Ligand '{ligand_name}' has no hydrogens ({target_ligand.height} atoms). Attempting protonation...")
            try:
                target_ligand_h = self._add_hydrogens_to_ligand(target_ligand, ligand_name)
                xyz_lig_raw = _to_xyz(target_ligand_h, "")
            except Exception:
                xyz_lig_raw = _to_xyz(target_ligand, "")
                
            lines = xyz_lig_raw.splitlines()
            n_lig_final = int(lines[0])
            xyz_lig_body = "\n".join(lines[2:])
        else:
            n_h = target_ligand.filter(pl.col("element") == "H").height
            print(f"  Ligand '{ligand_name}' already has {n_h} hydrogens ({target_ligand.height} atoms total)")
            xyz_lig_raw = _to_xyz(target_ligand, "")
            lines = xyz_lig_raw.splitlines()
            n_lig_final = int(lines[0])
            xyz_lig_body = "\n".join(lines[2:])
            
        # --- COMBINE ---
        total_atoms = n_prot_final + n_lig_final
        xyz_complex = f"{total_atoms}\nComplex_{ligand_name}\n{xyz_prot_body}\n{xyz_lig_body}"
        
        # Component XYZs for single points
        xyz_protein = f"{n_prot_final}\nProtein\n{xyz_prot_body}"
        xyz_ligand = f"{n_lig_final}\nLigand\n{xyz_lig_body}"
        
        return xyz_complex, xyz_protein, xyz_ligand, n_prot_final

    def _run_single_point(self, xyz_content: str, charge: int = 0, uhf: int = 0, solvent: str = "water") -> float:
        """Runs a single point energy calculation (no optimization)."""
        if shutil.which("xtb") is None:
            return 0.0

        run_id = str(uuid.uuid4())[:8]
        run_dir = self.work_dir / f"sp_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        xyz_file = run_dir / "system.xyz"
        with open(xyz_file, "w") as f:
            f.write(xyz_content)
            
        cmd = [
            "xtb", "system.xyz", 
            "--gfn", "2", 
            "--chrg", str(charge),
            "--uhf", str(uhf)
        ]
        if solvent:
            cmd.extend(["--alpb", solvent])
            
        energy = 0.0
        try:
            process = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
            if process.returncode != 0:
                # Log error but don't crash
                # print(f"xTB SP Error: {process.stderr}") 
                pass
            else:
                for line in process.stdout.splitlines():
                    if "TOTAL ENERGY" in line and "Eh" in line:
                        parts = line.split()
                        try:
                            idx = parts.index("Eh")
                            energy = float(parts[idx-1])
                        except: pass
        except:
            pass
        finally:
            if not self.keep_files:
                try:
                    shutil.rmtree(run_dir)
                except: pass
                
        return energy

    def _estimate_charge(self, pocket_atoms: pl.DataFrame) -> int:
        """
        Estimates the formal charge of the protein pocket based on residue composition.
        Heuristic:
          ASP, GLU: -1
          LYS, ARG: +1
          HIS: 0 (neutral at pH 7.4 usually)
        """
        charge = 0
        
        # We need residue_name and residue_seq to distinguish unique residues
        # If columns missing, return 0
        if "residue_name" not in pocket_atoms.columns:
            return 0
            
        # Group by unique residues (chain + seq + name) to count each residue once
        # Note: 'residue_seq' might be string or int.
        cols = ["residue_name"]
        if "residue_seq" in pocket_atoms.columns:
            cols.append("residue_seq")
        if "chain_id" in pocket_atoms.columns:
            cols.append("chain_id")
            
        unique_residues = pocket_atoms.unique(subset=cols)
        
        for row in unique_residues.iter_rows(named=True):
            res = row["residue_name"].upper()
            if res in ["ASP", "GLU"]:
                charge -= 1
            elif res in ["LYS", "ARG"]:
                charge += 1
            # HIS is usually neutral at pH 7.4 (pKa ~6.0)
            
        return charge

    def run_scoring(self, xyz_complex: str, xyz_protein: str, xyz_ligand: str, 
                    n_protein_atoms: int, charge: int = 0, 
                    uhf: int = 0, solvent: str = "water",
                    pocket_atoms: Optional[pl.DataFrame] = None,
                    save_structures: bool = False,
                    structure_id: str = "unknown") -> Dict[str, float]:
        """
        Runs GFN2-xTB optimization on the complex (freezing protein) and calculates
        interaction energy: E_int = E_complex - (E_protein + E_ligand).
        
        Args:
            save_structures: If True, saves input/output structures to debug_structures/
            structure_id: Identifier for saved debug files.
        
        Returns:
            Dictionary with results: 
            {"energy": float, "gap": float, "interaction_energy": float}
        """
        if shutil.which("xtb") is None:
            raise RuntimeError("'xtb' executable not found. Please install xtb.")

        # Auto-estimate charge if not provided and we have atom data
        if charge == 0 and pocket_atoms is not None:
            estimated_charge = self._estimate_charge(pocket_atoms)
            if estimated_charge != 0:
                print(f"  Estimated system charge: {estimated_charge}")
                charge = estimated_charge

        run_id = str(uuid.uuid4())[:8]
        run_dir = self.work_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        xyz_file = run_dir / "system.xyz"
        with open(xyz_file, "w") as f:
            f.write(xyz_complex)
            
        # Save debug structures if requested
        if save_structures or self.keep_files:
            self.debug_dir.mkdir(exist_ok=True, parents=True)
            debug_input = self.debug_dir / f"{structure_id}_input.xyz"
            shutil.copy(xyz_file, debug_input)
            
            # Convert to PDB for easier viewing if obabel exists
            if shutil.which("obabel"):
                debug_pdb = self.debug_dir / f"{structure_id}_input.pdb"
                subprocess.run(["obabel", str(debug_input), "-O", str(debug_pdb)], 
                             capture_output=True)
            
        # Create constraint input for xTB
        const_file = run_dir / "constrain.inp"
        with open(const_file, "w") as f:
            f.write(f"$fix\n atoms: 1-{n_protein_atoms}\n$end\n")
            
        # 1. Optimize Complex (Ligand flexible, Protein fixed)
        cmd = [
            "xtb", "system.xyz", 
            "--opt", 
            "--gfn", "2", 
            "--chrg", str(charge),
            "--uhf", str(uhf),
            "--input", "constrain.inp"
        ]
        
        if solvent:
            cmd.extend(["--alpb", solvent])
            
        results = {}
        
        try:
            process = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
            
            # Save optimized structure if requested
            xtbopt_file = run_dir / "xtbopt.xyz"
            if (save_structures or self.keep_files) and xtbopt_file.exists():
                debug_opt = self.debug_dir / f"{structure_id}_optimized.xyz"
                shutil.copy(xtbopt_file, debug_opt)
                
                if shutil.which("obabel"):
                    debug_opt_pdb = self.debug_dir / f"{structure_id}_optimized.pdb"
                    subprocess.run(["obabel", str(debug_opt), "-O", str(debug_opt_pdb)], 
                                 capture_output=True)
            
            if process.returncode != 0:
                print(f"xTB Error in {run_dir}:")
                # Print only the last few lines of stderr to avoid spam
                err_lines = process.stderr.splitlines()
                print("\n".join(err_lines[-5:]))
                return {}

            # Parse Complex Energy
            output = process.stdout
            for line in output.splitlines():
                if "TOTAL ENERGY" in line and "Eh" in line:
                    parts = line.split()
                    try:
                        idx = parts.index("Eh")
                        results["energy"] = float(parts[idx-1])
                    except: pass
                
                if "HOMO-LUMO GAP" in line:
                    parts = line.split()
                    try:
                        idx = parts.index("eV")
                        results["gap"] = float(parts[idx-1])
                    except: pass

            # 2. Calculate Single Point Energies for Components
            # We assume the protein energy is constant (since it was frozen)
            # We calculate ligand energy in vacuum (or solvent) to get interaction
            
            # E_protein (Single Point)
            # Note: Protein charge is tricky. We assume total charge - ligand charge?
            # For simplicity, we run SP on protein with same settings as complex
            # Ideally user should provide charges for components.
            # Here we assume charge is distributed or 0 for simplicity.
            e_protein = self._run_single_point(xyz_protein, charge=charge, uhf=uhf, solvent=solvent)
            
            # E_ligand (Single Point)
            # Ligand charge is usually 0 unless specified. 
            # If complex charge is 0, and protein is neutral, ligand is 0.
            e_ligand = self._run_single_point(xyz_ligand, charge=0, uhf=0, solvent=solvent)
            
            if "energy" in results and e_protein != 0.0 and e_ligand != 0.0:
                # Interaction Energy = E_complex - (E_protein + E_ligand)
                # Convert Hartree to kcal/mol (1 Eh = 627.5 kcal/mol)
                e_int_hartree = results["energy"] - (e_protein + e_ligand)
                results["interaction_energy"] = e_int_hartree * 627.509
                results["e_complex"] = results["energy"]
                results["e_protein"] = e_protein
                results["e_ligand"] = e_ligand

        except Exception as e:
            print(f"xTB execution failed: {e}")
        finally:
            if not self.keep_files:
                try:
                    shutil.rmtree(run_dir)
                except Exception:
                    pass
                    
        return results
