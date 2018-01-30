class Container:
    def __init__(self, layer):
        self.layers = []
        self._traverse_back(layer)
        self.layers.reverse()
        self.layers_name = [layer.name for layer in self.layers]

    def _traverse_back(self, layer):
        if layer not in self.layers:
            self.layers.append(layer)
        for prev in layer.prev_layers:
            siblings = set(prev.next_layers)
            # print(layer.name, [s.name for s in siblings])
            if not siblings.issubset(set(self.layers)):
                continue
            self._traverse_back(prev)
