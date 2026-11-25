import unittest
import numpy as np

from analise_frequencias.espectro import spectrum, EDGES, BANDS
from analise_frequencias.explorar import train_rows


class SpectrumTests(unittest.TestCase):
    def test_constant_and_frequency_grid(self):
        self.assertIsNone(spectrum(np.full((32, 32), 0.5)))
        signal = np.cos(2*np.pi*4*np.arange(32)/32)[None, :] * np.ones((32, 1))
        result = spectrum(signal)
        self.assertAlmostEqual(result["radius"][16, 20], .125)
        self.assertAlmostEqual(result["radius"][16, 12], .125)
        peaks = set(map(tuple, np.argwhere(result["power"] > result["power"].max() * .99)))
        self.assertEqual(peaks, {(16, 12), (16, 20)})
        self.assertAlmostEqual(result["features"]["Etotal"], 0.0, places=12)
        self.assertAlmostEqual(result["normalized"].sum(), 1.0)
        self.assertIsNone(result["radial_mean"][0])
        self.assertEqual(result["counts"][0], 0)

    def test_known_radial_frequency_and_energy(self):
        x, y = np.meshgrid(np.arange(32), np.arange(32))
        image = np.cos(2*np.pi*(6*x + 8*y)/32)
        result = spectrum(image)
        self.assertAlmostEqual(result["radius"][24, 22], .3125)
        self.assertGreater(result["features"]["E2"], .99999999)
        self.assertAlmostEqual(result["features"]["E1"], 0.0, places=12)
        self.assertAlmostEqual(result["features"]["E3"], 0.0, places=12)
        self.assertAlmostEqual(sum(result["features"][f"E{i}"] for i in (1, 2, 3)),
                               result["features"]["Etotal"], places=12)
        self.assertAlmostEqual(sum(result["radial_energy"][1:]), 1.0, places=12)
        self.assertTrue(np.isfinite(result["normalized"]).all())

    def test_bands_disjoint_coverage_and_nyquist(self):
        result = spectrum(np.cos(np.pi*np.arange(32))[None, :] * np.ones((32, 1)))
        radius = result["radius"]
        masks = [(radius >= a) & ((radius <= b) if i == 2 else (radius < b))
                 for i, (a,b) in enumerate(BANDS)]
        self.assertTrue(np.all(sum(m.astype(int) for m in masks) <= 1))
        self.assertTrue(np.array_equal(np.logical_or.reduce(masks), (radius >= .25) & (radius <= .5)))
        self.assertAlmostEqual(result["features"]["E3"], 1.0)
        self.assertEqual(radius[16, 0], .5)
        self.assertEqual(result["normalized"][0, 0], 0.0)
        self.assertEqual(len(EDGES), 17)

    def test_only_train_is_selected(self):
        rows = [{"experimental_split": x} for x in ("train", "validation", "test")]
        self.assertEqual(train_rows(rows), rows[:1])


if __name__ == "__main__":
    unittest.main()
