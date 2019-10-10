import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
import os
# import matplotlib.pyplot as plt
from util import util


if __name__ == '__main__':
	opt = TrainOptions().parse()
	netDtype = opt.netD
	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	save_dir = os.path.join(opt.checkpoints_dir, opt.name)

	model = create_model(opt)
	model.setup(opt)
	total_steps = 0

	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0

		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			if total_steps % opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time

			total_steps += opt.batch_size
			epoch_iter += opt.batch_size
			model.set_input(data)
			model.optimize_parameters()

			if total_steps % opt.print_freq == 0:
				losses = model.get_current_losses()
				t = (time.time() - iter_start_time) / opt.batch_size

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
							(epoch, total_steps))
				model.save_networks('latest')

			iter_data_time = time.time()
		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' %
						(epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)
			visuals = model.get_current_visuals()
			for label, im_data in visuals.items():
				im = util.tensor2im(im_data)
				img_name = 'image_%s_%s.png' % (epoch, label)
				img_loc = os.path.join(save_dir, img_name)
				util.save_image(im, img_loc)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
					(epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
		model.update_learning_rate()