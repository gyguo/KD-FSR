import logging
import datetime
logger = logging.getLogger(__name__)


class AttrDict(dict):
    """
    Subclass dict and define getter-setter.
    This behaves as both dict and obj.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


__C = AttrDict()
config = __C

__C.TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
__C.GPU_ID = [0, 1]
__C.NUM_WORKERS = 15
__C.DATASET = ''
__C.DATADIR = ''
__C.DISP_FREQ = 10  # frequency to display
__C.SEED = 0

# cudnn related setting
__C.CUDNN = AttrDict()
__C.CUDNN.BENCHMARK = False
__C.CUDNN.DETERMINISTIC = True
__C.CUDNN.ENABLE = True

# ## Model options
__C.MODEL = AttrDict()
__C.MODEL.TYPE = ''  # 'BASELINE' or 'FSR'

# ## Data options
__C.DATA = AttrDict()
__C.DATA.NUM_CLASSES = 20
__C.DATA.INPUT_SIZE = 112
__C.DATA.LARGE_SIZE = 448
__C.DATA.MIDDLE_SIZE = 224
__C.DATA.SMALL_SIZE = 112

# ## solver options
__C.SOLVER = AttrDict()
__C.SOLVER.START_LR = 0.001
__C.SOLVER.LR_STEPS = [30, 45]
__C.SOLVER.LR_FACTOR = 0.1
__C.SOLVER.NUM_EPOCHS = 50
__C.SOLVER.WEIGHT_DECAY = 5e-4
__C.SOLVER.MUMENTUM = 0.9

# ## Training options.
__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 1
# we use batch multipy technique, proposed by David Morton in
# https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672
__C.TRAIN.BATCH_MULTIPLY = 1

# ## Testing options.
__C.TEST = AttrDict()
__C.TEST.BATCH_SIZE = 1

# ## configs for baseline
__C.BASE = AttrDict()
__C.BASE.ARCH = ''
__C.BASE.TRAIN = AttrDict()
__C.BASE.TRAIN.BATCH_SIZE = 16
__C.BASE.TRAIN.BATCH_MULTIPLY = 1

__C.BASE.TEST = AttrDict()
__C.BASE.TEST.BATCH_SIZE = 16
__C.BASE.TEST.STATE_DIR = ''

__C.BASE.SOLVER = AttrDict()
__C.BASE.SOLVER.START_LR = 0.001
__C.BASE.SOLVER.LR_STEPS = [30, 45]
__C.BASE.SOLVER.LR_FACTOR = 0.1
__C.BASE.SOLVER.NUM_EPOCHS = 50
__C.BASE.SOLVER.WEIGHT_DECAY = 5e-4
__C.BASE.SOLVER.MUMENTUM = 0.9

# ## configs for FSR
__C.FSR = AttrDict()
__C.FSR.GROUPS = 64
__C.FSR.LARGE_ARCH = ''
__C.FSR.SMALL_ARCH = ''
__C.FSR.TEMP = 4
__C.FSR.ALPHA = 0.9
__C.FSR.FEATURE_TRIM = [0, 1]
__C.FSR.IN_PLANES = 512
__C.FSR.OUT_PLANES = 2048
# __C.FSR.LARGE_MODELDICT = ''
# __C.FSR.SMALL_MODELDICT = ''
__C.FSR.TRAIN = AttrDict()
__C.FSR.TRAIN.BATCH_SIZE = 1
__C.FSR.TRAIN.BATCH_MULTIPLY = 1
__C.FSR.TRAIN.LOSS_WEIGHTS = [1, 1]

__C.FSR.TEST = AttrDict()
__C.FSR.TEST.BATCH_SIZE = 1
__C.FSR.TEST.STATE_DIR = ''

__C.FSR.SOLVER = AttrDict()
__C.FSR.SOLVER.START_LR = 0.01
__C.FSR.SOLVER.LR_STEPS = [30, 45]
__C.FSR.SOLVER.LR_FACTOR = 0.1
__C.FSR.SOLVER.NUM_EPOCHS = 50
__C.FSR.SOLVER.WEIGHT_DECAY = 5e-4
__C.FSR.SOLVER.MUMENTUM = 0.9

# ## configs for KD
__C.KD = AttrDict()
__C.KD.ARCH_T = ''
__C.KD.ARCH_S = ''
__C.KD.MODELDICT_T = ''
__C.KD.TEMP = 4 # temperature hyper-parameter
__C.KD.ALPHA = 0.9
__C.KD.BETA = 1
__C.KD.TRAIN = AttrDict()
__C.KD.TRAIN.BATCH_SIZE = 1
__C.KD.TRAIN.BATCH_MULTIPLY = 1
__C.KD.TRAIN.LOSS_WEIGHTS = [1, 1]

__C.KD.TEST = AttrDict()
__C.KD.TEST.BATCH_SIZE = 1
__C.KD.TEST.STATE_DIR = ''

__C.KD.SOLVER = AttrDict()
__C.KD.SOLVER.START_LR = 0.01
__C.KD.SOLVER.LR_STEPS = [30, 45]
__C.KD.SOLVER.LR_FACTOR = 0.1
__C.KD.SOLVER.NUM_EPOCHS = 50
__C.KD.SOLVER.WEIGHT_DECAY = 5e-4
__C.KD.SOLVER.MUMENTUM = 0.9
def merge_dicts(dict_a, dict_b):
    from ast import literal_eval
    for key, value in dict_a.items():
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # The types must match, too.
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
                raise ValueError(
                    'Type mismatch ({} vs. {}) for config key: {}'.format(
                        type(dict_b[key]), type(value), key)
                )
        # Recursively merge dicts.
        if isinstance(value, AttrDict):
            try:
                merge_dicts(dict_a[key], dict_b[key])
            except BaseException:
                raise Exception('Error under config key: {}'.format(key))
        else:
            dict_b[key] = value


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen))
    merge_dicts(yaml_config, __C)


def cfg_from_list(args_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(args_list) % 2 == 0, 'Specify values or keys for args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # Handle the case when v is a string literal.
            val = literal_eval(value)
        except BaseException:
            val = value
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(val), type(cfg[subkey]))
        cfg[subkey] = val