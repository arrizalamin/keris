import numpy as np
from time import time
from tqdm import trange
from keris.container import Container


class Trainer(Container):
    def compile(self, loss_fn, optimizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._init_optimizer_params()

        self.epoch = 0
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

    def _init_optimizer_params(self):
        for layer in self.layers:
            if not layer.trainable:
                continue
            for key, param in layer.params.items():
                param_name = layer.name + key
                self.optimizer.add_parameter(param_name, param.shape)

    def _init_session(self):
        self.forward_outputs = {name: None for name in self.layers_name}
        self.backward_outputs = {
            (layer.name): {
                'dout': None,
                'traversed': False
            } for layer in self.layers}
        self.grads = {
            node.name: None for node in self.layers if node.trainable}

    def _all_prev_nodes_traversed(self, node):
        for prev_node in node.prev_layers:
            if self.forward_outputs[prev_node.name] is None:
                return False
        return True

    def _traverse_graph_forward(self, node, x, mode):
        if self.forward_outputs[node.name] is not None:
            return
        self.forward_outputs[node.name] = out = node.forward(x, mode)
        for layer in node.next_layers:
            if not self._all_prev_nodes_traversed(layer):
                continue

            if len(layer.prev_layers) > 1:
                out = []
                for prev in layer.prev_layers:
                    out.append(self.forward_outputs[prev.name])

            self._traverse_graph_forward(layer, out, mode)

    def _forward(self, X, y=None, mode='train', stop=None):
        self._init_session()
        # Perform forward pass
        self._traverse_graph_forward(self.layers[0], X, mode)
        last_layer_name = self.layers[-1].name
        out = self.forward_outputs[last_layer_name]

        if y is None:
            return out

        acc = (np.argmax(out, axis=1) == y).mean()
        loss, dout = self.loss_fn(out, y)

        return loss, acc, dout

    def _all_next_nodes_traversed(self, node):
        for next_node in node.next_layers:
            if not self.backward_outputs[next_node.name]['traversed']:
                return False
        return True

    def _traverse_graph_backward(self, node, x, mode):
        dout, grads = node.backward(x, mode)
        self.backward_outputs[node.name]['traversed'] = True

        for i, prev in enumerate(node.prev_layers):
            if type(dout) is tuple:
                prevDout = dout[i]
            else:
                prevDout = dout
            if self.backward_outputs[prev.name]['dout'] is None:
                self.backward_outputs[prev.name]['dout'] = prevDout
            else:
                self.backward_outputs[prev.name]['dout'] += prevDout

        if node.trainable:
            self.grads[node.name] = grads

        for i, layer in enumerate(node.prev_layers):
            if not self._all_next_nodes_traversed(layer):
                continue

            dout = self.backward_outputs[layer.name]['dout']

            self._traverse_graph_backward(layer, dout, mode)

    def _backward(self, dout, mode='train'):
        # Perform backpropagation
        self._traverse_graph_backward(self.layers[-1], dout, mode)
        # print(self.grads)

        for name, grad in self.grads.items():
            node = next(filter(lambda x: x.name == name, self.layers))
            self._update_param(node, name, grad)

        return dout

    def _update_param(self, node, namespace, grads):
        for key, grad in grads.items():
            param_name = namespace + key
            param = node.params[key]
            next_param = self.optimizer.update(param_name, param, grad)
            node.update_param(key, next_param)

    def train_on_batch(self, x, y):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Compute loss and gradient
        loss, acc, dout = self._forward(x, y, mode='train')
        self._backward(dout)

        return loss, acc

    def predict(self, data, stop=None):
        y = self._forward(data, mode='test', stop=stop)
        return y

    def _fit(self, batch, epochs=10, checkpoint=None, callback=None):
        te = trange(epochs)
        current_time = time()
        min_loss = 9999

        train_steps = batch.train_steps
        val_steps = batch.val_steps
        for epoch in te:
            ts = trange(train_steps)
            t_loss = 0
            t_acc = 0
            for t in ts:
                x_batch, y_batch = batch.get_batch('train', t)
                x_batch = (x_batch / 127.5) - 1
                step_loss, step_acc = self.train_on_batch(x_batch, y_batch)
                ts.set_description('train: loss=%g, acc=%g' %
                                   (step_loss, step_acc))
                t_loss += step_loss
                t_acc += step_acc
            t_loss /= train_steps
            t_acc /= train_steps
            self.metrics['train_loss'].append(t_loss)
            self.metrics['train_acc'].append(t_acc)

            self.epoch += 1
            self.optimizer.decrease_lr()

            acc = 0
            loss = 0
            for t in range(val_steps):
                x_batch, y_batch = batch.get_batch('validation', t)
                x_batch = (x_batch / 127.5) - 1

                val_loss, val_acc, _ = self._forward(
                    x_batch, y_batch, mode='test')
                min_loss = min(min_loss, val_loss)
                if val_loss <= min_loss and checkpoint is not None:
                    save_name = '%s-%d-%d' % (checkpoint, epoch, current_time)
                    self.save(save_name)
                acc += val_acc
                loss += val_loss
            acc /= val_steps
            loss /= val_steps
            self.metrics['val_loss'].append(loss)
            self.metrics['val_acc'].append(acc)
            te.set_description('loss: %g acc:%g' % (loss, acc))

            if callback is not None:
                params = self._get_parameters()
                callback(self.epoch, params, self.metrics['train_loss'],
                         self.metrics['val_loss'], self.metrics['train_acc'],
                         self.metrics['val_acc'])

        # print new line to prevent next prompt override progress bar
        print()
