from easydict import EasyDict
from typing import List, Optional, Union, Dict
import logging

logger = logging.getLogger(__name__)

def merge_configs(config_list: Optional[List[Union[Dict, EasyDict]]]) -> Optional[EasyDict]:
    if config_list == None or len(config_list) == 0:
        return None

    base_config = config_list[0]
    if type(base_config) is dict:
        base_config = EasyDict(base_config)

    if type(base_config) is not EasyDict:
        logger.error("The argument given to 'merge_configs' have to be of type dict or EasyDict.")
        return None

    for i in range(len(config_list) - 1):
        config_to_merge = config_list[i+1]
        if type(config_to_merge) is dict:
            config_to_merge = EasyDict(config_to_merge)
        _merge_add_a_into_b(config_to_merge, base_config)
    return base_config


def _merge_add_a_into_b(a: EasyDict, b: EasyDict) -> None:
    """
    Merge config dictionary a into config dictionary b,
    clobbering the options in b whenever they are also specified in a.
    New options that are only in a will be added to b.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # if the key from a is new to b simply add it
        if not k in b:
            b[k] = v
            continue

        # the types must match
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_add_a_into_b(a[k], b[k])
            except:
                logger.error(f'Error under config key: {k}')
                raise
        else:
            b[k] = v
