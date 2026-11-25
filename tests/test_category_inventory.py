import unittest
from genimage_prep.category_inventory import inventory


class CategoryInventoryTests(unittest.TestCase):
    def test_unconfirmed_mapping_and_unknown_evaluation_labels(self):
        classes = {str(i): [f"n{i:08}", f"class_{i}"] for i in range(1000)}
        train = {"stable_diffusion_v_1_4": {
            "train/nature/n00000003_1.JPEG": 0,
            "train/ai/003_sdv4_1.png": 1,
        }}
        test = {"stable_diffusion_v_1_4": {
            "test/nature/ILSVRC2012_val_00000001.JPEG": 0,
            "test/ai/003_sdv4_2.png": 1,
        }}
        rows, unknown = inventory(classes, train, test)
        self.assertEqual(len(rows), 2000)
        self.assertEqual(rows[6]["real_count"], 1)
        self.assertEqual(rows[6]["synthetic_prefix_count"], 1)
        self.assertEqual(rows[6]["genimage_index_confirmed"], "")
        self.assertEqual(rows[6]["balanced_pairs_confirmed"], "")
        self.assertEqual(rows[7]["real_count"], "")
        self.assertEqual(unknown["test"], 1)


if __name__ == "__main__":
    unittest.main()
