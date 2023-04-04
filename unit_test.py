import torch
import unittest
from blocks import SmartConv

"""
Test for SmartConv
"""
class TestSmartConv(unittest.TestCase):
    def setUp(self):
        # create an instance of the neural network
        target_height = 6
        target_width = 6
        in_channels = 1
        max_out_channels = 1
        kernel_size = 3
        self.net = SmartConv(in_channels, max_out_channels, target_height, target_width, kernel_size)

    def test_forward(self):
        # test the forward method of the neural network
        input = torch.ones(1, 1, 3, 3)
        output = self.net.forward(input)
        x_padding = torch.tensor([[[[1., 0., 1., 0., 1., 0.],
          [0., 0., 0., 0., 0., 0.],
          [1., 0., 1., 0., 1., 0.],
          [0., 0., 0., 0., 0., 0.],
          [1., 0., 1., 0., 1., 0.],
          [0., 0., 0., 0., 0., 0.]]]])

        self.assertTrue(torch.equal(self.net.x_padding, x_padding))
        self.assertTrue(torch.all(self.net.x_conv != 0))
        self.assertEqual(self.net.x_conv.size(), torch.Size([1, 1, 3, 3]))
        self.assertEqual(output.size(), torch.Size([1, 1, 6, 6]))

if __name__ == '__main__':
    unittest.main()
