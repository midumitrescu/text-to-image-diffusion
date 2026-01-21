import unittest


class DependenciesChecksCase(unittest.TestCase):

    def test_torch_installed(self):
        try:
            import torch
        except ImportError:
            self.fail("PyTorch is not installed")

    def test_cuda_available(self):
        import torch
        if not torch.cuda.is_available():
            self.fail("CUDA is not available")

    def test_cuda_version(self):
        import torch
        self.assertEqual("12.8", torch.version.cuda)

    def test_transformers_installed(self):
        try:
            import transformers
        except ImportError:
            self.fail("Transformers is not installed")

    def test_transformers_version_must_be_at_least_4(self):
        import transformers

        self.assertGreaterEqual(int(transformers.__version__.split(".")[0]), 4, "Transformers version must be 4+")

    def test_diffusers_installed(self):
        try:
            import diffusers
        except ImportError:
            self.fail("Diffusers is not installed")

    def test_diffusers_version_must_be_at_least_(self):
        import diffusers
        self.assertGreaterEqual(int(diffusers.__version__.split(".")[1]), 20, "Diffusers version must be 0.20+")





if __name__ == '__main__':
    unittest.main()
