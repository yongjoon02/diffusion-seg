import autorootcwd

class Registry():
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix # ex) UNet_model
        
        assert name not in self._obj_map, f"{name} is already registered in {self._name} registry."
        
        self._obj_map[name] = obj
        # print(f"Registering {name} to {self._name} registry.")  # 로그 제거

    def register(self, name:str=None, obj=None, suffix=None):
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                register_name = name if name is not None else func_or_class.__name__
                self._do_register(register_name, func_or_class, suffix)
                return func_or_class
            return deco
        
        # used as a function call
        if name is None:
            name = obj.__name__
        self._do_register(name, obj, suffix)
    
    def get(self, name, suffix='basicsr'):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f"Name {name} is not found, use name: {name}_{suffix}")
        if ret is None:
            raise KeyError(f"No object named {name} found in {self._name} registry.")
        return ret
    
    
    def __contains__(self, name):
        return name in self._obj_map
    
    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset')
ARCHS_REGISTRY = Registry('archs')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')