# API Reference

## Sicifus

::: sicifus.Sicifus
    handler: python
    options:
      members:
        - ingest
        - load
        - backbone
        - all_atom
        - ligands
        - get_structure
        - get_all_atoms
        - get_ligands
        - load_metadata
        - meta
        - meta_columns
        - hist
        - scatter
        - align_all
        - get_aligned_structure
        - generate_tree
        - cluster
        - annotate_clusters
        - tree_branch_lengths
        - tree_stats
        - clusters
        - get_cluster
        - get_cluster_for
        - get_cluster_siblings
        - cluster_summary
        - get_clustered_ids
        - analyze_ligand_binding
        - analyze_pi_stacking
        - analyze_ligand_contacts
        - get_binding_pockets
        - analyze_binding_pocket
        - repair_structure
        - calculate_stability
        - mutate_structure
        - load_mutations
        - mutate_batch
        - calculate_binding_energy
        - alanine_scan
        - position_scan
        - per_residue_energy

## Mutation Engine

::: sicifus.MutationEngine
    handler: python
    options:
      members:
        - repair
        - calculate_stability
        - mutate
        - load_mutations
        - mutate_batch
        - calculate_binding_energy
        - alanine_scan
        - position_scan
        - per_residue_energy

## Mutation

::: sicifus.Mutation
    handler: python
    options:
      members:
        - from_str
        - label

## Analysis Toolkit

::: sicifus.analysis.AnalysisToolkit
    handler: python
    options:
      members:
        - compute_rmsd_matrix
        - cluster_fast
        - build_tree
        - build_phylo_tree
        - cluster_from_tree
        - plot_tree
        - plot_circular_tree
        - build_similarity_network

## Ligand Analyzer

::: sicifus.analysis.LigandAnalyzer
    handler: python
    options:
      members:
        - find_binding_residues
        - plot_binding_histogram
        - detect_pi_stacking
        - plot_pi_stacking
        - find_ligand_atom_contacts
        - plot_ligand_contacts
        - build_ligand_mol
        - plot_ligand_2d
        - get_pocket_residues
        - plot_binding_pocket_composition

## CIF Loader

::: sicifus.io.CIFLoader
    handler: python
    options:
      members:
        - ingest_folder
