from .base_options import BaseOptions


class TestOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
		parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
		parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
		parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
		#  Dropout and Batchnorm has different behavioir during training and test.
		parser.add_argument('--eval', type=str, default=False, help='use eval mode during test time.')
		parser.add_argument('--num_test', type=int, default=100, help='how many test images to generate')

		# parser.set_defaults(model='test')
		# To avoid cropping, the loadSize should be the same as fineSize
		parser.set_defaults(loadSize=parser.get_default('fineSize'))
		self.isTrain = False
		return parser
