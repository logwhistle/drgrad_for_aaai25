import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Dropout
from tensorflow.python.ops.init_ops import Zeros, Ones, glorot_normal_initializer as glorot_normal
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops

class PredictionLayer(Layer):
    """
    Arguments
    - **task**: str, ``"binary"`` for binary logloss or ``"regression"`` for regression loss
    - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def activation_fn(name='mish'):
    fns = {}
    fns['relu'] = lambda x : tf.nn.relu(x)
    fns['tanh'] = lambda x : tf.nn.tanh(x)
    fns['swish'] = lambda x : tf.nn.swish(x)
    fns['mish'] = lambda x : x * tf.tanh(tf.nn.softplus(x))
    return fns[name]


def normalize(x):
    return x / (tf.norm(x, axis=1, keepdims=True) + 1e-8)

def cosine(x, y, keepdims=True):
    return tf.reduce_sum(tf.multiply(
            normalize(x), normalize(y),
            ), axis=1, keepdims=keepdims)

def mydiv(x, y):
    a = tf.reduce_sum(normalize(x), axis=1, keepdims=False)
    b = tf.reduce_sum(normalize(y), axis=1, keepdims=False)
    return tf.clip_by_value(tf.div_no_nan(a, b), 0.1, 5.0)

# import mymodel.global_var as global_var
import global_var
global_var._init()
def EXP(features, labels, mode):
    def _get_loss_fn():
        weight_boost_pos = 3.0
        weight_boost_neg = 1.0

        def fn(labels, logits):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            ones = tf.ones_like(labels, dtype=tf.dtypes.float32)
            loss *= tf.where(labels <= 0, ones * weight_boost_neg, ones * weight_boost_pos)
            return loss

        return fn if mode == tf.estimator.ModeKeys.TRAIN else None


    global learning_rate
    global optimizer
    learning_rate = global_var.get_value('learning_rate')
    optimizer = global_var.get_value('optimizer')
    def dnn_model(prefix, inputs, hidden_units, act):
        with tf.variable_scope(prefix) as scope:
            net = inputs
            for layer_id, dnn_hidden_unit in enumerate(hidden_units):
                if dnn_hidden_unit == 2:
                    kernel_initializer = tf.orthogonal_initializer(2 ** 0.5)
                else:
                    kernel_initializer = tf.glorot_uniform_initializer()
                net = tf.layers.dense(
                    inputs=net,
                    units=dnn_hidden_unit,
                    activation=None,
                    kernel_initializer=kernel_initializer)

                if layer_id < len(hidden_units) - 1:
                    net = act(net)
            return net

    hidden0 = 128
    hidden = 128
    sparses = []
    for i in range(32):
        name = 'sparse%d'%i
        dim = features[name].shape[1].value
        sparses.append(activation_fn('mish')(dnn_model('sparse_inputs%d'%i, features[name], [dim//2, dim], activation_fn('mish'))))
    sparse = dnn_model('sparse_tower', tf.concat(sparses, -1), [hidden0//2, hidden0], activation_fn('mish'))
    sparse = activation_fn('mish')(sparse)
    dense = dnn_model('dense_inputs', features['dense'], [hidden0//4], activation_fn('mish'))
    dense = activation_fn('mish')(dense)

    new = tf.concat([dense, sparse], -1)
    shared = dnn_model('input_sparse', new, [hidden0], activation_fn('mish'))
    nn_input = activation_fn('mish')(shared) # add special embedding: such as engage.. / add act func

    # layer1
    expert_share = dnn_model('expert_share', nn_input, [hidden], None)
    expert_share = activation_fn('mish')(expert_share)

    expert1 = dnn_model('expert1', nn_input, [hidden], None)
    expert1 = activation_fn('mish')(expert1)
    expert_input1 = tf.stack([expert1, expert_share], axis=1)
    gateNN1 = dnn_model('gateNN1', nn_input, [hidden, 2], activation_fn('mish'))
    gateNN1 = tf.nn.softmax(gateNN1)
    gateNN1 = tf.expand_dims(gateNN1, axis=-1)
    head_inputs1 = tf.reduce_sum(gateNN1 * expert_input1, axis=1, keep_dims=False)

    expert2 = dnn_model('expert2', nn_input, [hidden], None)
    expert2 = activation_fn('mish')(expert2)
    expert_input2 = tf.stack([expert2, expert_share], axis=1)
    gateNN2 = dnn_model('gateNN2', nn_input, [hidden, 2], activation_fn('mish'))
    gateNN2 = tf.nn.softmax(gateNN2)
    gateNN2 = tf.expand_dims(gateNN2, axis=-1)
    head_inputs2 = tf.reduce_sum(gateNN2 * expert_input2, axis=1, keep_dims=False)

    expert_input3 = tf.stack([expert1, expert2, expert_share], axis=1)
    gateNN3 = dnn_model('gateNN3', nn_input, [hidden, 3], activation_fn('mish'))
    gateNN3 = tf.nn.softmax(gateNN3)
    gateNN3 = tf.expand_dims(gateNN3, axis=-1)
    head_inputs3 = tf.reduce_sum(gateNN3 * expert_input3, axis=1, keep_dims=False)


    # layer 2
    expert_share2 = dnn_model('expert_share2', head_inputs3, [hidden], None)
    expert_share2 = activation_fn('mish')(expert_share2)

    expert21 = dnn_model('expert21', head_inputs1, [hidden], None)
    expert21 = activation_fn('mish')(expert21)
    expert_input21 = tf.stack([expert21, expert_share2], axis=1)
    gateNN21 = dnn_model('gateNN21', head_inputs1, [hidden, 2], activation_fn('mish'))
    gateNN21 = tf.nn.softmax(gateNN21)
    gateNN21 = tf.expand_dims(gateNN21, axis=-1)
    head_inputs21 = tf.reduce_sum(gateNN21 * expert_input21, axis=1, keep_dims=False)

    expert22 = dnn_model('expert22', head_inputs2, [hidden], None)
    expert22 = activation_fn('mish')(expert22)
    expert_input22 = tf.stack([expert22, expert_share2], axis=1)
    gateNN22 = dnn_model('gateNN22', head_inputs2, [hidden, 2], activation_fn('mish'))
    gateNN22 = tf.nn.softmax(gateNN22)
    gateNN22 = tf.expand_dims(gateNN22, axis=-1)
    head_inputs22 = tf.reduce_sum(gateNN22 * expert_input22, axis=1, keep_dims=False)


    multi_heads, logits = [], {}
    # share1 = dnn_model('share1', head_inputs3, [32, 1], activation_fn('relu'))
    # share2 = dnn_model('share1', head_inputs3, [32, 1], activation_fn('relu'))
    logits['salary'] = dnn_model('head1', head_inputs21, [hidden//2, 1], activation_fn('relu'))
    logits['marital_stat'] = dnn_model('head2', head_inputs22, [hidden//2, 1], activation_fn('relu'))
    multi_heads.append(tf.contrib.estimator.binary_classification_head(name='salary', loss_fn=_get_loss_fn(), weight_column="salaryWeight"))
    multi_heads.append(tf.contrib.estimator.binary_classification_head(name='marital_stat', weight_column="marital_statWeight"))
    heads = tf.contrib.estimator.multi_head(multi_heads)

    def _train_op_fn(loss):
        train_ops = []
        temp = []
        global learning_rate
        global optimizer
        print('optimizer is: ', optimizer)
        print('learning_rate is: ', learning_rate)
        global_step = tf.train.get_global_step()
        dnn_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        loss += tf.losses.get_regularization_loss()
        if optimizer == 'adam':
            dnn_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.9999)
        elif optimizer == 'adgrad':
            dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=0.08)
        else:
            dnn_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        gradients = dnn_optimizer.compute_gradients(loss, var_list=dnn_var)
        cache = {}
        for grad, var in gradients:
            head_vars = 'head1/dense/kernel:0'
            if var.name == head_vars:
                cache['head1'] = [tf.reshape(grad, [-1, hidden*hidden//2])]
                temp.append(grad)
            head_vars = 'head2/dense/kernel:0'
            if var.name == head_vars:
                cache['head2'] = [tf.reshape(grad, [-1, hidden*hidden//2])]
                temp.append(grad)


        flag_value, weight = tf.reduce_mean(cosine(cache['head1'], cache['head2'], False)), tf.reduce_mean(mydiv(cache['head1'], cache['head2']))
        flag = tf.where(flag_value <= 0, [0.0], [1.0])
        flag = tf.cast(flag, tf.float64)
        flag_value = tf.cast(flag_value, tf.float64)
        weight = tf.cast(weight, tf.float64)
        flag_value2, weight2 = tf.reduce_mean(cosine(cache['head2'], cache['head1'], False)), tf.reduce_mean(mydiv(cache['head2'], cache['head1']))
        flag2 = tf.where(flag_value2 <= 0, [0.0], [1.0])
        flag2 = tf.cast(flag2, tf.float64)
        flag_value2 = tf.cast(flag_value2, tf.float64)
        weight2 = tf.cast(weight2, tf.float64)
        for i, (grad, var) in enumerate(gradients):
            head_vars = 'head1/dense/kernel:0'
            if var.name == head_vars:
                grad += (temp[1]-(1.0-flag)*flag_value*temp[1])*weight
                gradients[i] = (grad, var)
            head_vars = 'head2/dense/kernel:0'
            if var.name == head_vars:
                grad += (temp[0]-(1.0-flag2)*flag_value2*temp[0])*weight2
                gradients[i] = (grad, var)

        train_ops.append(dnn_optimizer.apply_gradients(gradients))

        train_op = control_flow_ops.group(*train_ops)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        tf.logging.info('update_ops : %s' % update_ops)
        with tf.control_dependencies(update_ops):
            with tf.control_dependencies([train_op]):
                return state_ops.assign_add(global_step, 1).op

    return heads.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            # optimizer=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.9999),
            # optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.002),
            logits=logits,
            train_op_fn=_train_op_fn)