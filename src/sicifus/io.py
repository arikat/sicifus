import gemmi
import polars as pl
from pathlib import Path
import os
import shutil
import subprocess
import uuid
import numpy as np
from typing import List, Tuple, Optional

class CIFLoader:
    """
    Handles ingestion of CIF files into Polars DataFrames.
    """

    def __init__(self):
        pass

    def ingest_folder(self, input_folder: str, output_folder: str, batch_size: int = 100, 
                      file_extension: str = "cif", protonate: bool = False):
        """
        Ingests all structure files in a folder and saves them as a partitioned Parquet dataset.
        
        Args:
            input_folder: Path to the folder containing structure files.
            output_folder: Path to the folder where Parquet files will be saved.
            batch_size: Number of structures to process before writing a partition.
            file_extension: Extension of files to ingest (e.g., "cif" or "pdb").
            protonate: If True, uses PDBFixer (OpenMM) to add hydrogens to the structure 
                       before parsing. This is slower but ensures consistent protonation 
                       for energy calculations.
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for backbone, heavy_atoms, hydrogens, and ligands
        backbone_dir = output_path / "backbone"
        heavy_atom_dir = output_path / "heavy_atoms"
        hydrogens_dir = output_path / "hydrogens"
        ligands_dir = output_path / "ligands"
        
        backbone_dir.mkdir(exist_ok=True)
        heavy_atom_dir.mkdir(exist_ok=True)
        hydrogens_dir.mkdir(exist_ok=True)
        ligands_dir.mkdir(exist_ok=True)

        # Handle both .ext and .ext.gz
        files = list(input_path.glob(f"*.{file_extension}")) + list(input_path.glob(f"*.{file_extension}.gz"))
        print(f"Found {len(files)} {file_extension} files.")

        backbone_buffer = []
        heavy_atom_buffer = []
        hydrogens_buffer = []
        ligand_buffer = []
        
        batch_counter = 0
        
        for i, file_path in enumerate(files):
            try:
                # Parse structure (optionally protonating first)
                backbone_df, heavy_atom_df, hydrogens_df, ligand_df = self._parse_structure(file_path, protonate=protonate)
                
                if backbone_df is not None:
                    backbone_buffer.append(backbone_df)
                if heavy_atom_df is not None:
                    heavy_atom_buffer.append(heavy_atom_df)
                if hydrogens_df is not None:
                    hydrogens_buffer.append(hydrogens_df)
                if ligand_df is not None:
                    ligand_buffer.append(ligand_df)
                
                # Write batch if buffer is full or it's the last file
                if (len(backbone_buffer) >= batch_size) or (i == len(files) - 1):
                    self._write_batch(backbone_buffer, backbone_dir, batch_counter)
                    self._write_batch(heavy_atom_buffer, heavy_atom_dir, batch_counter)
                    self._write_batch(hydrogens_buffer, hydrogens_dir, batch_counter)
                    self._write_batch(ligand_buffer, ligands_dir, batch_counter)
                    
                    backbone_buffer = []
                    heavy_atom_buffer = []
                    hydrogens_buffer = []
                    ligand_buffer = []
                    print(f"Processed {i + 1}/{len(files)} files.")
                    batch_counter += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    def _df_to_pdb(self, df: pl.DataFrame) -> str:
        """
        Converts a DataFrame of atoms to a PDB formatted string.
        Ensures strict column alignment for OpenBabel compatibility.
        """
        lines = []
        for i, row in enumerate(df.iter_rows(named=True)):
            atom_name = str(row.get('atom_name', 'X')).strip()
            res_name = str(row.get('residue_name', 'UNK')).strip()[:3]
            chain_id = str(row.get('chain', 'A')).strip()[:1] # Truncate to 1 char
            res_seq = row.get('residue_number', 1)
            try:
                res_seq = int(res_seq)
            except:
                res_seq = 1
                
            x, y, z = row['x'], row['y'], row['z']
            elem = str(row.get('element', atom_name[0])).strip().upper()
            
            # Atom Name Alignment Logic (PDB Standard)
            if len(atom_name) >= 4:
                aname_fmt = f"{atom_name[:4]}"
            elif len(elem) == 2: # 2-letter element starts at 13
                aname_fmt = f"{atom_name:<4}"
            else: # 1-letter element starts at 14
                aname_fmt = f" {atom_name:<3}"

            line = (f"ATOM  {i+1:>5} {aname_fmt:<4} {res_name:<3} {chain_id:>1}{res_seq:>4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2}")
            lines.append(line)
        return "\n".join(lines)

    def _add_hydrogens_to_ligand(self, ligand_df: pl.DataFrame, ligand_name: str) -> pl.DataFrame:
        """
        Uses Meeko (preferred), RDKit, or OpenBabel to add hydrogens to the ligand.
        Returns a new DataFrame with hydrogens added.
        """
        meeko_success = False
        new_df = None
        
        # Use the first row of original DF as template for metadata
        template = ligand_df.row(0, named=True)
        
        # --- Helper: build RDKit mol from DataFrame coordinates ---
        def _build_mol_from_df(df):
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
            
            # Strategy 1: Full bond perception
            bonds_ok = False
            try:
                from rdkit.Chem import rdDetermineBonds
                rdDetermineBonds.DetermineConnectivity(mol)
                rdDetermineBonds.DetermineBondOrders(mol)
                bonds_ok = mol.GetNumBonds() > 0
            except Exception:
                pass
            
            # Strategy 2: Connectivity only
            if not bonds_ok:
                try:
                    from rdkit.Chem import rdDetermineBonds
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
            
            # Strategy 3: Distance-based fallback (0.8-1.85 Å = bonded)
            if not bonds_ok:
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
            
            return mol.GetMol(), names, bonds_ok
        
        # Strip existing hydrogens first
        ligand_heavy = ligand_df.filter(pl.col("element") != "H")
        if ligand_heavy.height == 0:
            return ligand_df
        
        n_heavy = ligand_heavy.height

        # --- Helper to convert mol_h → DataFrame ---
        def _mol_to_df(mol_h, atom_names=None):
            conf_h = mol_h.GetConformer()
            new_rows = []
            heavy_idx = 0
            h_counter = 0
            for i, atom in enumerate(mol_h.GetAtoms()):
                pos = conf_h.GetAtomPosition(i)
                elem = atom.GetSymbol()
                # Preserve original names for heavy atoms, generate H1, H2... for hydrogens
                if atom.GetAtomicNum() > 1 and atom_names and heavy_idx < len(atom_names):
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
                    "x": pos.x, "y": pos.y, "z": pos.z,
                    "b_factor": template["b_factor"],
                    "element": elem
                })
            return pl.DataFrame(new_rows)
        
        # 1. Try Meeko first
        try:
            from meeko import MoleculePreparation
            from rdkit import Chem
            
            mol, atom_names, bonds_ok = _build_mol_from_df(ligand_heavy)
            
            if not bonds_ok:
                raise ValueError("No bonds detected, cannot use Meeko")
            
            preparator = MoleculePreparation(merge_these_atom_types=())
            setups = preparator.prepare(mol)
            
            if setups:
                mol_h = setups[0].mol
                n_h_new = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
                
                if n_h_new == 0:
                    raise ValueError("Meeko added no hydrogens")
                
                if mol_h.GetNumConformers() > 0:
                    positions = mol_h.GetConformer().GetPositions()
                    if not (np.all(positions == 0.0) or np.any(np.isnan(positions))):
                        meeko_success = True
                        new_df = _mol_to_df(mol_h, atom_names)

        except ImportError:
            pass
        except Exception as e:
            pass

        if meeko_success and new_df is not None:
            return new_df

        # 2. Try RDKit (Fallback)
        rdkit_success = False
        new_df = None
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            try:
                _ = mol.GetNumAtoms()
            except (NameError, AttributeError):
                mol, atom_names, bonds_ok = _build_mol_from_df(ligand_heavy)
            
            mol_h = Chem.AddHs(mol, addCoords=True)
            n_h_added = mol_h.GetNumAtoms() - mol.GetNumAtoms()
            
            if n_h_added == 0:
                raise ValueError("RDKit added no hydrogens")
            
            # Refine Hydrogen Positions
            try:
                if mol_h.GetNumConformers() > 0:
                    conf = mol_h.GetConformer()
                    coord_map = {}
                    for atom in mol_h.GetAtoms():
                        if atom.GetAtomicNum() > 1:
                            idx = atom.GetIdx()
                            pos = conf.GetAtomPosition(idx)
                            coord_map[idx] = pos
                    AllChem.EmbedMolecule(mol_h, coordMap=coord_map, forceTol=0.01, useRandomCoords=True)
            except Exception:
                pass
            
            conf_h = mol_h.GetConformer()
            positions = conf_h.GetPositions()
            
            if not (np.all(positions == 0.0) or np.any(np.isnan(positions))):
                rdkit_success = True
                new_df = _mol_to_df(mol_h, atom_names)
                
        except ImportError:
            pass
        except Exception as e:
            pass

        if rdkit_success and new_df is not None:
            return new_df

        # 3. Fallback to OpenBabel
        if shutil.which("obabel"):
            try:
                pdb_content = self._df_to_pdb(ligand_df)
                
                run_id = str(uuid.uuid4())[:8]
                temp_in = f"temp_lig_{run_id}.pdb"
                temp_out = f"temp_lig_{run_id}.xyz"
                
                with open(temp_in, "w") as f:
                    f.write(pdb_content)
                    
                cmd = ["obabel", temp_in, "-O", temp_out, "-h", "-p", "7.4"]
                subprocess.run(cmd, capture_output=True, check=True)
                
                if os.path.exists(temp_out):
                    with open(temp_out, "r") as f:
                        lines = f.readlines()
                    try:
                        n_atoms = int(lines[0])
                        new_rows = []
                        for line in lines[2:]:
                            parts = line.split()
                            if len(parts) >= 4:
                                elem = parts[0]
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                new_rows.append({
                                    "structure_id": template["structure_id"],
                                    "model": template["model"],
                                    "chain": template["chain"],
                                    "residue_name": template["residue_name"],
                                    "residue_number": template["residue_number"],
                                    "atom_name": elem,
                                    "x": x, "y": y, "z": z,
                                    "b_factor": template["b_factor"],
                                    "element": elem
                                })
                        if len(new_rows) > 0:
                            try:
                                os.remove(temp_in)
                                os.remove(temp_out)
                            except: pass
                            return pl.DataFrame(new_rows)
                    except ValueError:
                        pass
                        
                try:
                    if os.path.exists(temp_in): os.remove(temp_in)
                    if os.path.exists(temp_out): os.remove(temp_out)
                except: pass
                
            except Exception as e:
                try:
                    if os.path.exists(temp_in): os.remove(temp_in)
                    if os.path.exists(temp_out): os.remove(temp_out)
                except: pass

        return ligand_df

    def _parse_structure(self, file_path: Path, protonate: bool = False) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """
        Parses a single structure file (CIF or PDB) and extracts:
          - backbone: CA atoms only (fast alignment/RMSD)
          - heavy_atoms: All protein heavy atoms (contacts, pi-stacking)
          - hydrogens: Protein hydrogens only (energy scoring)
          - ligands: non-polymer, non-water atoms
        """
        temp_pdb = None
        try:
            parse_path = str(file_path)
            
            # --- PROTONATION STEP ---
            if protonate:
                try:
                    from pdbfixer import PDBFixer
                    from openmm.app import PDBFile
                    from openmm import Platform
                except ImportError:
                    print("Warning: PDBFixer/OpenMM not installed. Skipping protonation.")
                    # Fallback to normal parsing
                else:
                    try:
                        # PDBFixer can read PDB and PDBx/mmCIF
                        fixer = PDBFixer(filename=str(file_path))
                        
                        # Apply standard fixes
                        fixer.findMissingResidues()
                        fixer.findMissingAtoms()
                        fixer.addMissingAtoms()
                        fixer.addMissingHydrogens(7.4)
                        
                        # Write to temp PDB
                        import tempfile
                        fd, temp_pdb = tempfile.mkstemp(suffix=".pdb")
                        os.close(fd)
                        
                        with open(temp_pdb, 'w') as f:
                            PDBFile.writeFile(fixer.topology, fixer.positions, f)
                            
                        # Update parse path to the protonated PDB
                        parse_path = temp_pdb
                        
                    except Exception as e:
                        print(f"  Protonation failed for {file_path.name}: {e}")
                        # Fallback to original file
                        if temp_pdb and os.path.exists(temp_pdb):
                            os.remove(temp_pdb)
                        temp_pdb = None

            # --- GEMMI PARSING ---
            # Use gemmi to parse the file (either original or protonated temp PDB)
            structure = gemmi.read_structure(parse_path)
            
            # Only remove hydrogens if we DIDN'T ask to protonate
            # If protonate=True, we want to keep them!
            if not protonate:
                structure.remove_hydrogens()
            
            structure_id = file_path.name.split('.')[0]
            
            backbone_data = []
            heavy_atom_data = []
            hydrogens_data = []
            ligand_data = []
            
            for model in structure:
                # Defensive check for model.name
                model_name = getattr(model, 'name', '1')
                if not isinstance(model_name, str):
                    model_name = str(model_name)

                for chain in model:
                    for residue in chain:
                        # Check if it's a polymer residue
                        res_info = gemmi.find_tabulated_residue(residue.name)
                        is_amino_acid = res_info.is_amino_acid()
                        is_water = res_info.is_water()
                        
                        for atom in residue:
                            atom_data = {
                                "structure_id": structure_id,
                                "model": model_name,
                                "chain": chain.name,
                                "residue_name": residue.name,
                                "residue_number": str(residue.seqid),
                                "atom_name": atom.name,
                                "x": atom.pos.x,
                                "y": atom.pos.y,
                                "z": atom.pos.z,
                                "b_factor": atom.b_iso,
                                "element": atom.element.name
                            }
                            
                            if is_amino_acid:
                                if atom.element.name == "H":
                                    hydrogens_data.append(atom_data)
                                else:
                                    heavy_atom_data.append(atom_data)
                                    # CA atoms also go to backbone (for fast alignment)
                                    if atom.name == "CA":
                                        backbone_data.append(atom_data)
                            elif not is_water:
                                ligand_data.append(atom_data)

            backbone_df = pl.DataFrame(backbone_data) if backbone_data else None
            heavy_atom_df = pl.DataFrame(heavy_atom_data) if heavy_atom_data else None
            hydrogens_df = pl.DataFrame(hydrogens_data) if hydrogens_data else None
            ligand_df = pl.DataFrame(ligand_data) if ligand_data else None
            
            # --- LIGAND PROTONATION (RDKit) ---
            if protonate and ligand_df is not None and ligand_df.height > 0:
                try:
                    # Process each ligand separately to add hydrogens
                    protonated_ligands = []
                    # Get unique identifiers for ligands
                    unique_ligands = ligand_df.unique(subset=["chain", "residue_number", "residue_name"])
                    
                    for row in unique_ligands.iter_rows(named=True):
                        # Extract single ligand
                        sub_ligand = ligand_df.filter(
                            (pl.col("chain") == row["chain"]) & 
                            (pl.col("residue_number") == row["residue_number"]) &
                            (pl.col("residue_name") == row["residue_name"])
                        )
                        
                        # Add hydrogens
                        try:
                            # Use helper method
                            sub_ligand_h = self._add_hydrogens_to_ligand(sub_ligand, row["residue_name"])
                            protonated_ligands.append(sub_ligand_h)
                        except Exception:
                            # Fallback to original if RDKit fails
                            protonated_ligands.append(sub_ligand)
                            
                    if protonated_ligands:
                        ligand_df = pl.concat(protonated_ligands)
                        
                except Exception as e:
                    print(f"  Ligand protonation failed for {file_path.name}: {e}")

            # Cast columns to ensure schema consistency
            schema = {
                "structure_id": pl.Utf8,
                "model": pl.Utf8,
                "chain": pl.Utf8,
                "residue_name": pl.Utf8,
                "residue_number": pl.Utf8,
                "atom_name": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
                "b_factor": pl.Float64,
                "element": pl.Utf8
            }
            
            for df in [backbone_df, heavy_atom_df, hydrogens_df, ligand_df]:
                if df is not None:
                    # Only cast columns that exist
                    cast_cols = [pl.col(c).cast(t) for c, t in schema.items() if c in df.columns]
                    if cast_cols:
                        df = df.with_columns(cast_cols)

            return backbone_df, heavy_atom_df, hydrogens_df, ligand_df
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None, None, None, None
        finally:
            # Cleanup temp file
            if temp_pdb and os.path.exists(temp_pdb):
                try:
                    os.remove(temp_pdb)
                except: pass

    def _write_batch(self, df_list: List[pl.DataFrame], output_dir: Path, batch_index: int):
        """
        Concatenates a list of DataFrames and writes to a Parquet file.
        """
        if not df_list:
            return
            
        batch_df = pl.concat(df_list)
        output_file = output_dir / f"part_{batch_index}.parquet"
        batch_df.write_parquet(output_file)
