import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

from tools.TBSL.analysis._config import align_steps
from tools.TBSL.analysis.evaluate import emission_fields, evaluate, shell_volumes_m3, smooth_block
from tools.TBSL.analysis.section_7_4_table import TableCase, compute_case_metrics
from tools.TBSL.analysis.utils.parse_data import iter_ave_chunk_blocks, parse_ave_chunk_output, parse_global_output
from tools.TBSL.analysis.utils import sl_saha_emission as saha


class AnalysisChecks(unittest.TestCase):
    def write_temporary(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "input.txt"
        path.write_text(text)
        return path

    @unittest.skipUnless((FIXTURES / "historical_expected.json").exists(), "Optional historical research data are local only")
    def test_historical_table_metrics_unchanged(self):
        expected = json.loads((FIXTURES / "historical_expected.json").read_text())
        case = TableCase("1e6", 1e6, "1.0", "full", str(FIXTURES / "historical_global.table"), str(FIXTURES / "historical_radial.out"))
        actual = compute_case_metrics(case)
        for key, value in expected["table_metrics"].items():
            with self.subTest(metric=key):
                if isinstance(value, str):
                    self.assertEqual(actual[key], value)
                else:
                    np.testing.assert_allclose(actual[key], value, rtol=2e-14)

    @unittest.skipUnless((FIXTURES / "historical_expected.json").exists(), "Optional historical research data are local only")
    def test_historical_emission_matches_original_scripts(self):
        expected = json.loads((FIXTURES / "historical_expected.json").read_text())
        rows, _, _, _ = evaluate(FIXTURES / "historical_global.table", FIXTURES / "historical_radial.out", expected["ep_scale"], "paper")
        actual = [row["lambda_ge_200nm_mW"] for row in rows]
        np.testing.assert_allclose(actual, expected["band_mW"], rtol=2e-12, atol=1e-80)

    def test_all_nine_paper_recipes_import(self):
        recipes = json.loads((ROOT / "recipes.json").read_text())
        self.assertEqual(len(recipes), 9)
        for recipe in recipes.values():
            self.assertTrue(callable(importlib.import_module("tools.TBSL.analysis." + recipe["module"]).main))

    def test_saha_quadratic_residual_and_limits(self):
        temperature = np.geomspace(300, 1e10, 200)
        density = np.full(temperature.shape, 1e24)
        q = saha.compute_saha_ionization_fraction(temperature, density, 15.76)
        S = (2 * np.pi * saha.M_E * saha.K_B * temperature / saha.H_PLANCK**2)**1.5 * np.exp(-15.76 * saha.E_CHARGE / (saha.K_B * temperature))
        A = S / density
        # Normalized polynomial remains well conditioned as q -> 1.
        np.testing.assert_allclose((q*q + A*q - A) / np.maximum(A, 1e-300), 0, atol=8e-16)
        self.assertTrue(np.all(np.diff(q) >= 0))
        self.assertLess(q[0], 1e-100)
        self.assertGreater(q[-1], 0.999999)
        # Original subtractive root returns zero in this nearly fully ionized limit.
        high = saha.compute_saha_ionization_fraction(np.array([1e12]), np.array([1e5]), 15.76)
        np.testing.assert_allclose(high, 1.0, atol=1e-15)

    def test_number_and_mass_density_conversions(self):
        n_si = saha.number_density_from_lammps_number_real(np.array([0.0, 0.01]))
        np.testing.assert_allclose(n_si, [0, 1e28])
        mass_g_cm3 = 1e28 * 39.95 * saha.AMU_KG / 1000
        np.testing.assert_allclose(saha.number_density_from_lammps_mass_real(mass_g_cm3, 39.95), 1e28)

    def test_ep_scaling_and_zero_density(self):
        temp = np.array([15000., 30000., 0.])
        density = np.array([.01, .01, 0.])
        baseline = emission_fields(temp, density, 1)
        scaled = emission_fields(temp, density / 1e5, 1e5)
        for x, y in zip(baseline, scaled):
            np.testing.assert_allclose(x, y, rtol=1e-14)
        self.assertEqual(baseline[2][-1], 0)

    def test_band_integral_matches_spectral_kernel(self):
        # Substitute x=a/lambda: normalized band integral becomes integral exp(-x) dx.
        for temp in (3000., 15000., 1e5):
            xmax = saha.H_PLANCK * saha.C_LIGHT / (saha.K_B * temp * 200e-9)
            x = np.linspace(0, xmax, 10001)
            integral = (getattr(np, "trapezoid", None) or np.trapz)(np.exp(-x), x)
            np.testing.assert_allclose(saha.band_fraction_ge_lambda_closed_form(temp), integral, rtol=1e-6)

    def test_si_spherical_volume_and_wall_clipping(self):
        centers = np.array([.5, 1.5, 2.5]) * 1e4
        expected = 4 * np.pi / 3 * (3e-6)**3
        np.testing.assert_allclose(shell_volumes_m3(centers).sum(), expected, rtol=2e-15)
        np.testing.assert_allclose(shell_volumes_m3(centers, 1.8e4).sum(), 4*np.pi/3*(1.8e-6)**3, rtol=2e-15)

    def test_charge_and_normalized_spectrum_use_si_volumes(self):
        radii = np.array([.5,1.5]) * 1e4
        temperature = np.full((1,2), 20000.)
        density = np.full((1,2), 1e27)
        q, ne, total = saha.compute_saha_charge_distribution(temperature, density, radii, 15.76)
        np.testing.assert_allclose(total, ne @ shell_volumes_m3(radii), rtol=1e-15)
        wavelengths = np.geomspace(1., 1e8, 5000)
        power = np.full((1,2), 3e10)
        angstrom_result, _ = saha.normalized_spectrum_from_P_closed_form(power, temperature, radii, wavelengths)
        metre_result, _ = saha.normalized_spectrum_from_P_closed_form(power, temperature, radii*1e-10, wavelengths, length_unit_for_volume="m")
        np.testing.assert_allclose(angstrom_result, metre_result, rtol=1e-14)
        integrated = (getattr(np, "trapezoid", None) or np.trapz)(angstrom_result, wavelengths*1e-9, axis=1)
        np.testing.assert_allclose(integrated, power @ shell_volumes_m3(radii), rtol=5e-5)

    def test_scaling_profiles_align_to_distribution_sampling(self):
        from tools.TBSL.analysis.section_7_1_convergence import SweepCase, load_case
        # Extra global samples deliberately make the minimum-radius global index
        # larger than the number of distribution frames.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "global.table"
            radial = Path(directory) / "radial.out"
            dense = [[i,i,1000+i*100,30-i if i<6 else 26,1000+i*100,10] for i in range(7)]
            target.write_text("Step Time Temp v_SLradius c_coretemp v_pressbubble\n"+"\n".join(" ".join(map(str,row)) for row in dense)+"\n")
            text = "# Chunk Coord1 Ncount temp v_slvirial density/mass density/number\n"
            for step in (0,3,6):
                text += f"{step} 5 500\n"
                for j in range(5):
                    text += f"{j+1} {1+2*j} 100 {1000+100*step} -1 1 .01\n"
            radial.write_text(text)
            result = load_case(SweepCase("synthetic",1e6,"black",str(target),str(radial)))
            self.assertGreater(len(result["radial_temperature_rmin"]), 1)
            self.assertEqual(len(result["radius_time_ns"]), len(dense))

    def test_smoothing_preserves_counts_charge_and_integrated_densities(self):
        names = "Chunk Coord1 Ncount temp v_slcharge density/mass density/number".split()
        centers = np.array([.5, 1.5, 2.5, 3.5])
        volumes = 4*np.pi/3*np.diff(np.arange(5.)**3)
        counts = np.array([10., 30., 70., 5.])
        block = np.column_stack((np.arange(1,5), centers, counts, [100,200,400,900], [0,.1,.2,.3], 2*counts/volumes, counts/volumes))
        smoothed = smooth_block(0, names, block, 100)
        np.testing.assert_array_equal(smoothed[:,:3], block[:,:3])
        np.testing.assert_allclose(np.dot(smoothed[:,4], counts), np.dot(block[:,4], counts), rtol=3e-6)
        np.testing.assert_allclose(smoothed[:,5:] .T @ volumes, block[:,5:].T @ volumes, rtol=3e-6)
        # Preserve the original algorithm's small final wall group explicitly.
        self.assertEqual(smoothed[-1,3], 900)

    def test_moment_temperature_is_invariant_to_radial_boost(self):
        names = "Chunk Coord1 Ncount v_radial_velocity v_speed_squared density/number".split()
        for shift in (0., .1, -.2):
            vr = .04 + shift
            block = np.array([[1,.5,100,vr,vr*vr+.003,.01], [2,1.5,100,vr,vr*vr+.003,.01]])
            radial = self.write_temporary("# " + " ".join(names) + "\n0 2 200\n" + "\n".join(" ".join(map(str,row)) for row in block) + "\n")
            global_file = self.write_temporary("Step Time v_SLradius\n0 0 2\n")
            rows, _, _, _ = evaluate(global_file, radial, 10., "radial-flow")
            np.testing.assert_allclose(rows[0]["T_max_K"], 39.95e7/(3*8.31446261815324)*.003, rtol=2e-14)

    def test_global_log_repeated_headers_and_exact_duplicate(self):
        path = self.write_temporary("LAMMPS\nStep Time Temp\n0 0 300\nWARNING: example\n1 2 400\nLoop time of ...\nStep Time Temp\n1 2 400\n2 4 500\n")
        table, _ = parse_global_output(path)
        self.assertEqual(table["Time"], [0,2,4])

    def test_theory_table_accepts_generic_heading(self):
        path = self.write_temporary("frac_r Temp\n0 300\n1 400\n")
        table, _ = parse_global_output(path)
        self.assertEqual(table, {"frac_r": [0,1], "Temp": [300,400]})

    def test_conflicting_restart_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            parse_global_output(self.write_temporary("Step Time Temp\n0 0 300\n0 0 400\n"))

    def test_truncated_chunk_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            list(iter_ave_chunk_blocks(self.write_temporary("# Chunk Coord1 Ncount temp\n0 2 5\n1 .5 5 300\n")))

    def test_changing_radial_grid_rejected_by_fixed_grid_api(self):
        path = self.write_temporary("# Chunk Coord1 Ncount temp\n0 1 5\n1 .5 5 300\n1 1 5\n1 .6 5 300\n")
        with self.assertRaisesRegex(ValueError, "grid changes"):
            parse_ave_chunk_output(path)

    def test_out_of_range_or_unsorted_steps_rejected(self):
        for global_steps, radial_steps in (([0,1],[2]), ([0,2],[1]), ([1,0],[0])):
            with self.assertRaises(ValueError):
                align_steps(global_steps, radial_steps)

    def test_thermal_mode_never_silently_substitutes_paper_temperature(self):
        global_file = self.write_temporary("Step Time v_SLradius\n0 0 2\n")
        radial = self.write_temporary("# Chunk Coord1 Ncount temp density/number\n0 2 200\n1 .5 100 300 .01\n2 1.5 100 300 .01\n")
        with self.assertRaises(KeyError):
            evaluate(global_file, radial, 1., "radial-flow")


if __name__ == "__main__":
    unittest.main()
