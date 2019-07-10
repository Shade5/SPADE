import os.path
from data.pix2pix_dataset import Pix2pixDataset


class NingalooDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=14)
        parser.set_defaults(aspect_ratio=2.0)
        parser.set_defaults(no_instance=True)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def is_image_file(self, filename):
        return filename.endswith('png')

    def make_dataset_rec(self, dir, images):
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    def make_dataset(self, dir):
        images = []
        labels = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for episodes in os.listdir(dir):
            episode_path = os.path.join(dir, episodes, 'cameras')
            assert os.path.isdir(episode_path), '%s is not a valid directory' % dir
            for camera_name in os.listdir(episode_path):
                camera_path = os.path.join(episode_path, camera_name)
                if 'depth' in camera_name:
                    continue
                assert os.path.isdir(camera_path), '%s is not a valid directory' % dir
                for fname in os.listdir(camera_path):
                    assert self.is_image_file(fname), '%s is not a valid file' % fname
                    path = os.path.join(camera_path, fname)
                    if 'seg' in camera_name:
                        labels.append(path)
                    else:
                        images.append(path)
        return sorted(images), sorted(labels)

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        label_dir = os.path.join(root, phase)
        image_paths, label_paths = self.make_dataset(label_dir)

        return label_paths, image_paths, []

    def paths_match(self, path1, path2):
        path1_split = path1.split('/')
        path2_split = path2.split('/')
        # compare step number and episode name
        return path1_split[-1] == path2_split[-1] and path1_split[-3] == path2_split[-3] and path1_split[-4] == \
               path2_split[-4]
