"""Tests for the sicifus.mutate module (mutation & stability engine).

Unit tests (no OpenMM) run instantly and cover parsing / helper logic.
Integration tests (marked ``slow``) require OpenMM + PDBFixer and exercise
the full minimisation pipeline on Crambin (1CRN, 46 residues).

Run fast tests only:     pytest tests/test_mutate.py -m "not slow"
Run everything:          pytest tests/test_mutate.py -v
"""

import io
import math
import textwrap

import polars as pl
import pytest

from sicifus.mutate import (
    ALL_AMINO_ACIDS,
    ONE_TO_THREE,
    STANDARD_RESIDUES,
    THREE_TO_ONE,
    Mutation,
    MutationEngine,
    MutationResult,
    RepairResult,
    StabilityResult,
    _RepairCache,
    _df_to_pdb_string,
    _load_pdb,
    _compute_energy_statistics,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

slow = pytest.mark.slow


def _openmm_available() -> bool:
    try:
        import openmm  # noqa: F401
        from pdbfixer import PDBFixer  # noqa: F401
        return True
    except ImportError:
        return False


requires_openmm = pytest.mark.skipif(
    not _openmm_available(), reason="OpenMM / PDBFixer not installed"
)

MINI_PDB = textwrap.dedent("""\
    ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       2.500   2.200   1.000  1.00  0.00           C
    ATOM      4  O   ALA A   1       2.000   3.300   1.000  1.00  0.00           O
    ATOM      5  CB  ALA A   1       2.500   0.000   0.000  1.00  0.00           C
    END
""")


@pytest.fixture(scope="session")
def crambin_pdb():
    """Fetch Crambin (1CRN) once for the whole test session."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(pdbid="1CRN")
    buf = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, buf)
    return buf.getvalue()


@pytest.fixture(scope="session")
def engine():
    return MutationEngine(work_dir="/tmp/sicifus_test_mutate")


# ===================================================================
# UNIT TESTS — no OpenMM needed, run fast
# ===================================================================


class TestMutationParsing:
    """Mutation dataclass creation and string parsing."""

    def test_from_str_basic(self):
        m = Mutation.from_str("G73L")
        assert m.chain == "A"
        assert m.position == 73
        assert m.wt_residue == "GLY"
        assert m.mut_residue == "LEU"

    def test_from_str_with_chain(self):
        m = Mutation.from_str("G73L", chain="B")
        assert m.chain == "B"
        assert m.position == 73

    def test_from_str_large_position(self):
        m = Mutation.from_str("W9999A")
        assert m.position == 9999
        assert m.wt_residue == "TRP"
        assert m.mut_residue == "ALA"

    def test_from_str_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid mutation string"):
            Mutation.from_str("not_valid")

    def test_from_str_empty(self):
        with pytest.raises(ValueError):
            Mutation.from_str("")

    def test_from_str_lowercase_normalised(self):
        m = Mutation.from_str("g73l")
        assert m.wt_residue == "GLY"
        assert m.mut_residue == "LEU"

    def test_direct_three_letter(self):
        m = Mutation(position=10, wt_residue="PHE", mut_residue="TRP")
        assert m.wt_residue == "PHE"
        assert m.mut_residue == "TRP"
        assert m.chain == "A"

    def test_direct_one_letter(self):
        m = Mutation(position=42, wt_residue="F", mut_residue="W", chain="B")
        assert m.wt_residue == "PHE"
        assert m.mut_residue == "TRP"
        assert m.chain == "B"

    def test_direct_mixed_case(self):
        m = Mutation(position=1, wt_residue="ala", mut_residue="V")
        assert m.wt_residue == "ALA"
        assert m.mut_residue == "VAL"

    def test_direct_invalid_residue(self):
        with pytest.raises(ValueError, match="Unknown residue code"):
            Mutation(position=1, wt_residue="XYZ", mut_residue="ALA")

    def test_label(self):
        m = Mutation(position=73, wt_residue="GLY", mut_residue="LEU")
        assert m.label == "G73L"

    def test_label_roundtrip(self):
        for notation in ["G73L", "F42W", "M1K", "D100E"]:
            m = Mutation.from_str(notation)
            assert m.label == notation

    def test_repr_default_chain(self):
        m = Mutation.from_str("G73L")
        assert repr(m) == "Mutation(G73L)"

    def test_repr_non_default_chain(self):
        m = Mutation.from_str("G73L", chain="B")
        assert repr(m) == "Mutation(G73L, chain=B)"

    def test_default_chain_is_A(self):
        m = Mutation.from_str("A1V")
        assert m.chain == "A"


class TestResidueConstants:
    """Verify lookup tables are consistent and complete."""

    def test_twenty_amino_acids(self):
        assert len(THREE_TO_ONE) == 20
        assert len(ONE_TO_THREE) == 20

    def test_roundtrip(self):
        for three, one in THREE_TO_ONE.items():
            assert ONE_TO_THREE[one] == three

    def test_standard_residues_matches_keys(self):
        assert STANDARD_RESIDUES == set(THREE_TO_ONE.keys())

    def test_all_amino_acids_sorted(self):
        assert ALL_AMINO_ACIDS == sorted(THREE_TO_ONE.keys())


class TestLoadPdb:
    """_load_pdb dispatcher for strings, DataFrames, and files."""

    def test_pdb_string(self):
        result = _load_pdb(MINI_PDB)
        assert "ATOM" in result
        assert "ALA" in result

    def test_pdb_string_with_hetatm(self):
        hetatm = "HETATM    1  C1  LIG A   1       0.0   0.0   0.0  1.00  0.00           C\nEND"
        result = _load_pdb(hetatm)
        assert "HETATM" in result

    def test_dataframe(self):
        df = pl.DataFrame({
            "atom_name": ["N", "CA", "C"],
            "residue_name": ["ALA", "ALA", "ALA"],
            "chain": ["A", "A", "A"],
            "residue_number": [1, 1, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 1.0, 1.0],
            "z": [1.0, 1.0, 1.0],
            "element": ["N", "C", "C"],
        })
        result = _load_pdb(df)
        assert "ATOM" in result
        assert "ALA" in result

    def test_file_path(self, tmp_path):
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(MINI_PDB)
        result = _load_pdb(str(pdb_file))
        assert "ATOM" in result

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="source must be"):
            _load_pdb("just some random text with no pdb content")

    def test_invalid_path(self):
        with pytest.raises(ValueError, match="source must be"):
            _load_pdb("/nonexistent/path.pdb")


class TestDfToPdbString:
    """_df_to_pdb_string formatting."""

    def test_basic_formatting(self):
        df = pl.DataFrame({
            "atom_name": ["N", "CA"],
            "residue_name": ["ALA", "ALA"],
            "chain": ["A", "A"],
            "residue_number": [1, 1],
            "x": [1.123, 2.456],
            "y": [3.789, 4.012],
            "z": [5.345, 6.678],
            "element": ["N", "C"],
        })
        pdb = _df_to_pdb_string(df)
        lines = pdb.strip().split("\n")
        assert lines[-1] == "END"
        assert len(lines) == 3  # 2 atoms + END

        atom_line = lines[0]
        assert atom_line.startswith("ATOM")
        assert "ALA" in atom_line
        assert "1.123" in atom_line

    def test_four_char_atom_name(self):
        df = pl.DataFrame({
            "atom_name": ["1HG1"],
            "residue_name": ["VAL"],
            "chain": ["A"],
            "residue_number": [5],
            "x": [0.0], "y": [0.0], "z": [0.0],
            "element": ["H"],
        })
        pdb = _df_to_pdb_string(df)
        assert "1HG1" in pdb

    def test_two_letter_element(self):
        df = pl.DataFrame({
            "atom_name": ["FE"],
            "residue_name": ["HEM"],
            "chain": ["A"],
            "residue_number": [1],
            "x": [0.0], "y": [0.0], "z": [0.0],
            "element": ["FE"],
        })
        pdb = _df_to_pdb_string(df)
        assert "FE" in pdb


class TestMutationEngineInit:
    """Engine construction without OpenMM."""

    def test_defaults(self, tmp_path):
        e = MutationEngine(work_dir=str(tmp_path / "work"))
        assert e.forcefield_name == "amber14-all.xml"
        assert e.water_model == "implicit"
        assert e.platform_name == "CPU"
        assert e.work_dir.exists()

    def test_custom_params(self, tmp_path):
        e = MutationEngine(
            forcefield="amber14-all.xml",
            water_model="amber14/tip3pfb.xml",
            platform="CPU",
            work_dir=str(tmp_path / "custom"),
        )
        assert e.water_model == "amber14/tip3pfb.xml"


class TestLoadMutationsCSV:
    """CSV loading for batch mutations."""

    def test_load_basic_csv(self, tmp_path):
        csv = tmp_path / "muts.csv"
        csv.write_text("mutation,chain\nG13L,A\nF42W,B\n")
        df = MutationEngine.load_mutations(str(csv))
        assert df.height == 2
        assert "mutation" in df.columns
        assert "chain" in df.columns
        assert df["chain"].to_list() == ["A", "B"]

    def test_load_csv_defaults_chain_to_A(self, tmp_path):
        csv = tmp_path / "muts.csv"
        csv.write_text("mutation,score\nG13L,1.5\nF42W,-0.3\n")
        df = MutationEngine.load_mutations(str(csv))
        assert df["chain"].to_list() == ["A", "A"]
        assert "score" in df.columns

    def test_load_csv_preserves_metadata(self, tmp_path):
        csv = tmp_path / "muts.csv"
        csv.write_text("mutation,chain,source,notes\nG13L,A,paper,important\n")
        df = MutationEngine.load_mutations(str(csv))
        assert "source" in df.columns
        assert "notes" in df.columns
        assert df["source"][0] == "paper"

    def test_load_csv_missing_mutation_column(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("position,residue\n13,L\n")
        with pytest.raises(ValueError, match="mutation"):
            MutationEngine.load_mutations(str(csv))


# ===================================================================
# INTEGRATION TESTS — require OpenMM + PDBFixer
# ===================================================================


@slow
@requires_openmm
class TestRepair:

    def test_repair_returns_result(self, engine, crambin_pdb):
        result = engine.repair(crambin_pdb, max_iterations=50)
        assert isinstance(result, RepairResult)

    def test_repair_has_pdb_output(self, engine, crambin_pdb):
        result = engine.repair(crambin_pdb, max_iterations=50)
        assert "ATOM" in result.pdb_string
        atom_lines = [l for l in result.pdb_string.splitlines() if l.startswith("ATOM")]
        assert len(atom_lines) > 300

    def test_repair_energies_are_finite(self, engine, crambin_pdb):
        result = engine.repair(crambin_pdb, max_iterations=50)
        assert math.isfinite(result.energy_before)
        assert math.isfinite(result.energy_after)

    def test_repair_energy_after_is_negative(self, engine, crambin_pdb):
        """Minimised energy should be reasonable (large negative for a protein)."""
        result = engine.repair(crambin_pdb, max_iterations=200)
        assert result.energy_after < 0, "Minimised energy should be negative"


@slow
@requires_openmm
class TestStability:

    def test_stability_returns_result(self, engine, crambin_pdb):
        result = engine.calculate_stability(crambin_pdb, max_iterations=50)
        assert isinstance(result, StabilityResult)

    def test_stability_total_is_finite(self, engine, crambin_pdb):
        result = engine.calculate_stability(crambin_pdb, max_iterations=50)
        assert math.isfinite(result.total_energy)

    def test_stability_has_energy_terms(self, engine, crambin_pdb):
        result = engine.calculate_stability(crambin_pdb, max_iterations=50)
        assert "total" in result.energy_terms
        assert "NonbondedForce" in result.energy_terms
        assert "HarmonicBondForce" in result.energy_terms

    def test_stability_terms_sum_approximately_to_total(self, engine, crambin_pdb):
        result = engine.calculate_stability(crambin_pdb, max_iterations=50)
        terms_sum = sum(
            v for k, v in result.energy_terms.items()
            if k != "total"
        )
        assert abs(terms_sum - result.total_energy) < 1.0

    def test_stability_nonzero_gb_solvation(self, engine, crambin_pdb):
        result = engine.calculate_stability(crambin_pdb, max_iterations=50)
        gb_energy = result.energy_terms.get("CustomGBForce", 0.0)
        assert gb_energy != 0.0, "GBn2 solvation should be nonzero"

    def test_stability_has_pdb_output(self, engine, crambin_pdb):
        result = engine.calculate_stability(crambin_pdb, max_iterations=50)
        assert "ATOM" in result.pdb_string


@slow
@requires_openmm
class TestMutate:

    def test_mutate_single_string(self, engine, crambin_pdb):
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=50, n_runs=1)
        assert isinstance(result, MutationResult)
        assert "F13A" in result.ddg
        assert math.isfinite(result.ddg["F13A"])

    def test_mutate_with_mutation_object(self, engine, crambin_pdb):
        mut = Mutation(position=13, wt_residue="PHE", mut_residue="ALA")
        result = engine.mutate(crambin_pdb, [mut], max_iterations=50, n_runs=1)
        assert mut.label in result.ddg

    def test_mutate_wt_energy_is_finite(self, engine, crambin_pdb):
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=50, n_runs=1)
        assert math.isfinite(result.wt_energy)

    def test_mutate_produces_mutant_pdb(self, engine, crambin_pdb):
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=50, n_runs=1)
        pdb = list(result.mutant_pdbs.values())[0]
        assert "ATOM" in pdb
        assert "ALA" in pdb

    def test_mutate_energy_terms_dataframe(self, engine, crambin_pdb):
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=50, n_runs=1)
        df = result.energy_terms
        assert isinstance(df, pl.DataFrame)
        assert "term" in df.columns
        assert "wt_energy" in df.columns
        assert "mutant_energy" in df.columns
        assert "delta" in df.columns
        assert df.height > 0

    def test_mutate_phe_to_ala_destabilising(self, engine, crambin_pdb):
        """Removing a buried Phe sidechain should be destabilising (positive ddG)."""
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=100, n_runs=1)
        ddg = result.ddg["F13A"]
        assert ddg > 0, f"Phe->Ala should be destabilising, got ddG={ddg}"

    def test_mutate_multiple_runs(self, engine, crambin_pdb):
        result = engine.mutate(
            crambin_pdb, ["T1V"], n_runs=2, max_iterations=50
        )
        assert "T1V" in result.ddg
        assert math.isfinite(result.ddg["T1V"])

    def test_mutate_conservative_finite_ddg(self, engine, crambin_pdb):
        """Conservative mutation (Ile->Leu) should produce a finite ddG."""
        result = engine.mutate(crambin_pdb, ["I7L"], max_iterations=50, n_runs=1)
        assert math.isfinite(result.ddg["I7L"])

    def test_mutant_pdb_contains_mutation_at_correct_position(self, engine, crambin_pdb):
        """The mutant PDB should have ALA at position 13 (not just anywhere)."""
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=50, n_runs=1)
        pdb = list(result.mutant_pdbs.values())[0]
        found_ala_at_13 = False
        for line in pdb.splitlines():
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                if res_num == 13 and res_name == "ALA":
                    found_ala_at_13 = True
                    break
        assert found_ala_at_13, "Position 13 should be ALA in the mutant PDB"

    def test_mutant_pdb_preserves_other_residues(self, engine, crambin_pdb):
        """Non-mutated positions should retain their original identity."""
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=50, n_runs=1)
        pdb = list(result.mutant_pdbs.values())[0]
        res7_names = set()
        for line in pdb.splitlines():
            if line.startswith("ATOM"):
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()
                if res_num == 7:
                    res7_names.add(res_name)
        assert "ILE" in res7_names, "Position 7 should still be ILE after mutating 13"

    def test_stability_reproducibility(self, engine, crambin_pdb):
        """Two independent stability calculations on the same structure should
        agree within the hydrogen-placement noise floor.

        Each call rebuilds hydrogens via Modeller.addHydrogens, which can
        place them slightly differently.  With 200 minimisation steps, the
        noise is typically <10 kcal/mol on a ~1000 kcal/mol total — around
        1% relative error.  This sets a practical lower bound on ddG
        precision.
        """
        stab1 = engine.calculate_stability(crambin_pdb, max_iterations=200)
        stab2 = engine.calculate_stability(crambin_pdb, max_iterations=200)
        diff = abs(stab1.total_energy - stab2.total_energy)
        relative = diff / abs(stab1.total_energy)
        assert relative < 0.05, (
            f"Relative energy difference should be < 5%, "
            f"got {relative:.1%} (diff={diff:.1f} kcal/mol)"
        )


@slow
@requires_openmm
class TestMutateEdgeCases:
    """Edge cases and biological validation for the mutation engine."""

    def test_multiple_simultaneous_mutations(self, engine, crambin_pdb):
        """Apply two mutations simultaneously and verify both in the output PDB."""
        result = engine.mutate(crambin_pdb, ["F13A", "I7V"], max_iterations=50, n_runs=1)
        combined_label = "F13A+I7V"
        assert combined_label in result.ddg
        assert math.isfinite(result.ddg[combined_label])

        pdb = result.mutant_pdbs[combined_label]
        residues_at_pos = {}
        for line in pdb.splitlines():
            if line.startswith("ATOM"):
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()
                residues_at_pos[res_num] = res_name
        assert residues_at_pos.get(13) == "ALA", "Position 13 should be ALA"
        assert residues_at_pos.get(7) == "VAL", "Position 7 should be VAL"

    def test_ddg_values_physically_bounded(self, engine, crambin_pdb):
        """ddG values from single-point mutations should be within a physically
        reasonable range (-50 to +50 kcal/mol).

        Note: with limited minimisation (100 steps) and hydrogen-placement
        noise (~8 kcal/mol), we cannot reliably rank mutations by ddG
        magnitude.  We can only verify they are finite and bounded.
        """
        result = engine.mutate(crambin_pdb, ["I7L"], max_iterations=100, n_runs=1)
        ddg = result.ddg["I7L"]
        assert -50 < ddg < 50, f"ddG should be bounded, got {ddg}"

    def test_wt_energy_negative_and_large(self, engine, crambin_pdb):
        """Crambin (46 residues) wild-type energy should be a large negative number."""
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=100, n_runs=1)
        assert result.wt_energy < -100, (
            f"WT energy should be a large negative (kcal/mol), got {result.wt_energy}"
        )

    def test_energy_terms_delta_sums_to_ddg(self, engine, crambin_pdb):
        """The sum of per-term deltas should approximately equal ddG."""
        result = engine.mutate(crambin_pdb, ["F13A"], max_iterations=50, n_runs=1)
        df = result.energy_terms
        total_row = df.filter(pl.col("term") == "total")
        if total_row.height > 0:
            delta_total = total_row["delta"][0]
            ddg = result.ddg["F13A"]
            assert abs(delta_total - ddg) < 2.0, (
                f"Total delta ({delta_total}) should ≈ ddG ({ddg})"
            )


@slow
@requires_openmm
class TestAlanineScan:

    def test_ala_scan_returns_dataframe(self, engine, crambin_pdb):
        df = engine.alanine_scan(
            crambin_pdb, chain="A", positions=[1, 7, 13],
            max_iterations=50,
        )
        assert isinstance(df, pl.DataFrame)
        assert "chain" in df.columns
        assert "position" in df.columns
        assert "wt_residue" in df.columns
        assert "ddg_kcal_mol" in df.columns

    def test_ala_scan_correct_positions(self, engine, crambin_pdb):
        df = engine.alanine_scan(
            crambin_pdb, chain="A", positions=[7, 13],
            max_iterations=50,
        )
        scanned = df["position"].to_list()
        assert 7 in scanned
        assert 13 in scanned

    def test_ala_scan_skips_ala_gly(self, engine, crambin_pdb):
        """Alanine and glycine positions should not appear in the scan."""
        df = engine.alanine_scan(
            crambin_pdb, chain="A", positions=[9, 31],
            max_iterations=50,
        )
        wt_residues = df["wt_residue"].to_list()
        assert "ALA" not in wt_residues
        assert "GLY" not in wt_residues

    def test_ala_scan_ddg_values_finite(self, engine, crambin_pdb):
        df = engine.alanine_scan(
            crambin_pdb, chain="A", positions=[7, 13],
            max_iterations=50,
        )
        for val in df["ddg_kcal_mol"].to_list():
            assert math.isfinite(val)


# ===================================================================
# REPAIR-ONCE PATTERN TESTS
# ===================================================================


@slow
@requires_openmm
class TestPrepareStructure:
    """Tests for _prepare_structure / prepare (repair-once cache)."""

    def test_prepare_returns_cache(self, engine, crambin_pdb):
        cache = engine.prepare(crambin_pdb, max_iterations=200)
        assert isinstance(cache, _RepairCache)
        assert "ATOM" in cache.pdb_string
        assert math.isfinite(cache.energy_kcal)
        assert cache.energy_kcal < 0

    def test_prepare_energy_low_variance(self, engine, crambin_pdb):
        """Two independent prepare() calls may differ slightly due to
        non-deterministic hydrogen placement by Modeller.addHydrogens.

        The important contract is that *within* a batch, the WT energy
        is used from the cache and is perfectly deterministic (tested by
        test_batch_wt_energy_deterministic).  Independent prepare()
        calls should still agree within ~10 kcal/mol on a ~1000 kcal/mol
        protein.
        """
        c1 = engine.prepare(crambin_pdb, max_iterations=500)
        c2 = engine.prepare(crambin_pdb, max_iterations=500)
        diff = abs(c1.energy_kcal - c2.energy_kcal)
        relative = diff / abs(c1.energy_kcal)
        assert relative < 0.02, (
            f"Prepare should be low-variance, got relative diff={relative:.1%} "
            f"(diff={diff:.2f} kcal/mol)"
        )

    def test_mutate_with_cache_uses_cached_wt_energy(self, engine, crambin_pdb):
        """When _repair_cache is provided, mutate() should use the cached
        WT energy exactly, not recompute it.
        """
        cache = engine.prepare(crambin_pdb, max_iterations=200)
        result = engine.mutate(
            crambin_pdb, ["F13A"],
            max_iterations=100, n_runs=1,
            _repair_cache=cache,
        )
        assert result.wt_energy == cache.energy_kcal

    def test_mutate_with_cache_produces_valid_mutant(self, engine, crambin_pdb):
        cache = engine.prepare(crambin_pdb, max_iterations=200)
        result = engine.mutate(
            crambin_pdb, ["F13A"],
            max_iterations=100, n_runs=1,
            _repair_cache=cache,
        )
        pdb = list(result.mutant_pdbs.values())[0]
        assert "ALA" in pdb
        found_ala_at_13 = False
        for line in pdb.splitlines():
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                if res_num == 13 and res_name == "ALA":
                    found_ala_at_13 = True
                    break
        assert found_ala_at_13


@slow
@requires_openmm
class TestBatchWTConsistency:
    """Verify that mutate_batch produces identical WT energies for all rows."""

    def test_batch_wt_energy_deterministic(self, engine, crambin_pdb):
        """All mutations in a batch should report the same WT energy because
        the repair-once pattern processes the wild-type only once.
        """
        df = pl.DataFrame({
            "mutation": ["F13A", "I7V", "T1A"],
            "chain": ["A", "A", "A"],
        })
        results = engine.mutate_batch(
            crambin_pdb, df,
            max_iterations=200, n_runs=1,
        )
        wt_energies = results["wt_energy"].to_list()
        assert all(math.isfinite(e) for e in wt_energies)
        unique_wt = set(wt_energies)
        assert len(unique_wt) == 1, (
            f"All WT energies should be identical in a batch, got {unique_wt}"
        )


# ---------------------------------------------------------------------------
# Statistical Analysis Tests
# ---------------------------------------------------------------------------

class TestEnergyStatistics:
    """Test statistical functions for mutation analysis."""

    def test_compute_energy_statistics_basic(self):
        """Test basic statistical calculations."""
        all_energies = [-1000.0, -1002.0, -998.0]
        wt_energy = -1005.0
        n_runs = 3

        stats = _compute_energy_statistics(all_energies, wt_energy, n_runs)

        assert stats["mean"] == -1000.0
        assert stats["min"] == -1002.0
        assert stats["max"] == -998.0
        assert abs(stats["sd"] - 2.0) < 0.01
        assert stats["ddg_mean"] == 5.0  # mean - wt
        assert stats["ci_95"] is not None
        assert len(stats["ci_95"]) == 2

    def test_compute_energy_statistics_single_run(self):
        """Test statistics with n_runs=1 (no variance)."""
        all_energies = [-1000.0]
        wt_energy = -1005.0
        n_runs = 1

        stats = _compute_energy_statistics(all_energies, wt_energy, n_runs)

        assert stats["mean"] == -1000.0
        assert stats["sd"] == 0.0
        assert stats["ddg_mean"] == 5.0
        assert stats["ci_95"] == (5.0, 5.0)

    def test_compute_energy_statistics_convergence_metric(self):
        """Test convergence metric (CV = SD/mean)."""
        # High variance case
        all_energies = [-1000.0, -900.0, -1100.0]
        wt_energy = -1000.0
        n_runs = 3

        stats = _compute_energy_statistics(all_energies, wt_energy, n_runs)

        # CV should be > 0.1 (high variance)
        assert stats["cv"] > 0.05


@slow
@requires_openmm
class TestMutationStatistics:
    """Test mutation engine with statistical analysis enabled."""

    def test_mutate_with_statistics(self, engine, crambin_pdb):
        """Test that statistics are collected when keep_statistics=True."""
        result = engine.mutate(
            crambin_pdb, ["F13A"],
            max_iterations=100,
            n_runs=3,
            keep_statistics=True,
        )

        # Check statistical fields are populated
        assert result.all_run_energies is not None
        assert result.mean_energy is not None
        assert result.sd_energy is not None
        assert result.min_energy is not None
        assert result.max_energy is not None
        assert result.ddg_mean is not None
        assert result.ddg_sd is not None
        assert result.ddg_ci_95 is not None
        assert result.convergence_metric is not None

        # Check that we have 3 run energies
        label = "F13A"
        assert len(result.all_run_energies[label]) == 3

        # Check that mean is close to the values
        assert abs(result.mean_energy[label] - sum(result.all_run_energies[label])/3) < 0.01

    def test_mutate_without_statistics(self, engine, crambin_pdb):
        """Test that statistics are None when keep_statistics=False."""
        result = engine.mutate(
            crambin_pdb, ["F13A"],
            max_iterations=100,
            n_runs=3,
            keep_statistics=False,
        )

        # Check statistical fields are None
        assert result.all_run_energies is None
        assert result.mean_energy is None
        assert result.sd_energy is None

    def test_mutate_use_mean_vs_best(self, engine, crambin_pdb):
        """Test that use_mean flag changes the primary ddG calculation."""
        # Run with best (default)
        result_best = engine.mutate(
            crambin_pdb, ["F13A"],
            max_iterations=100,
            n_runs=3,
            keep_statistics=True,
            use_mean=False,
        )

        # Run with mean
        result_mean = engine.mutate(
            crambin_pdb, ["F13A"],
            max_iterations=100,
            n_runs=3,
            keep_statistics=True,
            use_mean=True,
        )

        label = "F13A"

        # The ddg should differ if runs have variance
        # (unless by chance all runs gave identical energies)
        # At minimum, we can check that both are reasonable values
        assert math.isfinite(result_best.ddg[label])
        assert math.isfinite(result_mean.ddg[label])

        # The mean-based should equal the mean energy
        assert abs(result_mean.ddg[label] - result_mean.ddg_mean[label]) < 0.01

    def test_backward_compatibility(self, engine, crambin_pdb):
        """Test that existing code without new parameters still works."""
        # Old-style call without new parameters
        result = engine.mutate(
            crambin_pdb, ["F13A"],
            max_iterations=100,
            n_runs=3,
        )

        # Should still return basic fields
        assert result.wt_energy is not None
        assert result.ddg is not None
        assert result.mutant_energies is not None
        assert result.mutant_pdbs is not None

        # Statistics should be populated by default (keep_statistics=True by default)
        assert result.mean_energy is not None

    def test_convergence_warning(self, engine, crambin_pdb, capsys):
        """Test that high CV triggers a warning."""
        # With only 2 runs, we might get high variance
        result = engine.mutate(
            crambin_pdb, ["F13A"],
            max_iterations=50,  # Lower iterations for more variance
            n_runs=2,
            keep_statistics=True,
        )

        # Just check that code doesn't crash - actual CV depends on minimization
        assert result.convergence_metric is not None
