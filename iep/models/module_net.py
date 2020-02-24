from iep.models.layers import ResidualBlock, GlobalAveragePool, Flatten
import iep.programs
import tensorflow as tf
from collections import OrderedDict


class ConcatBlock(tf.keras.Model):
    def __init__(self, dim, with_residual=True, with_batchnorm=True):
        super(ConcatBlock, self).__init__()
        self.proj = tf.keras.layers.Conv2D(dim, kernel_size=(1, 1), padding='same')
        self.res_block = ResidualBlock(dim, with_residual=with_residual,
                                       with_batchnorm=with_batchnorm)

    def __call__(self, x, y):
        print("Shape of x and y ", x.shape, y.shape)
        #x = tf.transpose(x, perm=[0, 2, 3, 1])
        #y = tf.transpose(y, perm=[0, 2, 3, 1])
        #print("Shape of x and y changed", x.shape, y.shape)
        out = tf.concat([x, y], 1)  # Concatentate along depth
        print("Shape of out changed before", out.shape)
        out = tf.transpose(out, perm=[0, 2, 3, 1])
        out = self.proj(out)
        print("Shape of out after proj", out.shape)
        out = tf.nn.relu(out)
        print("Shape of out after relu", out.shape)
        out = tf.transpose(out, perm=[0, 3, 1, 2])
        out = self.res_block(out)
        print("Shape of out changed", out.shape)
        return out


def build_stem(feature_dim, module_dim, num_layers=2, with_batchnorm=True):
    #print("dims : ", feature_dim, module_dim)
    layers = []
    prev_dim = feature_dim
    model = tf.keras.Sequential()
    for i in range(num_layers):
        model.add(tf.keras.layers.Conv2D(module_dim, kernel_size=(3, 3), padding='same'))
        if with_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        prev_dim = module_dim

    # for i in range(num_layers):
    #     layers.append(tf.keras.layers.Conv2D(prev_dim, module_dim, kernel_size=(3, 3), padding=1))
    #     if with_batchnorm:
    #         layers.append(tf.keras.layers.BatchNormalization())
    #     layers.append(tf.keras.layers.ReLU())
    #     prev_dim = module_dim
    #     print("Added Layer #", i)

    #model = tf.keras.Sequential(layers=layers)
    print("Model Created!")
    return model


def build_classifier(module_C, module_H, module_W, num_answers,
                     fc_dims=[], proj_dim=None, downsample='maxpool2',
                     with_batchnorm=True, dropout=0):
    layers = []
    prev_dim = module_C * module_H * module_W
    if proj_dim is not None and proj_dim > 0:
        layers.append(tf.keras.layers.Conv2D(proj_dim, kernel_size=(1, 1)))
        if with_batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.ReLU())
        prev_dim = proj_dim * module_H * module_W
    if downsample == 'maxpool2':
        layers.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
        prev_dim //= 4
    elif downsample == 'maxpool4':
        layers.append(tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=4))
        prev_dim //= 16
    layers.append(Flatten())
    for next_dim in fc_dims:
        layers.append(tf.keras.layers.Dense(next_dim, input_shape=(prev_dim, )))
        if with_batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.ReLU())
        if dropout > 0:
            layers.append(tf.keras.layers.Dropout(dropout))
        prev_dim = next_dim
    layers.append(tf.keras.layers.Dense(num_answers, input_shape=(prev_dim, )))
    model = tf.keras.Sequential(layers=layers)
    return model


class ModuleNet(tf.keras.Model):
    def __init__(self, vocab, feature_dim=(1024, 14, 14),
                 stem_num_layers=2,
                 stem_batchnorm=False,
                 module_dim=128,
                 module_residual=True,
                 module_batchnorm=False,
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 verbose=True):
        super(ModuleNet, self).__init__()
        self.modules = OrderedDict()
        print("Building stem ...")
        self.stem = build_stem(feature_dim[0], module_dim,
                               num_layers=stem_num_layers,
                               with_batchnorm=stem_batchnorm)

        if verbose:
            print('Here is my stem:')
            print(self.stem)

        num_answers = len(vocab['answer_idx_to_token'])
        module_H, module_W = feature_dim[1], feature_dim[2]
        print("Time to Build classifier")
        self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
                                           classifier_fc_layers,
                                           classifier_proj_dim,
                                           classifier_downsample,
                                           with_batchnorm=classifier_batchnorm,
                                           dropout=classifier_dropout)

        if verbose:
            print('Here is my classifier:')
            print(self.classifier)
        self.stem_times = []
        self.module_times = []
        self.classifier_times = []
        self.timing = False

        self.function_modules = {}
        self.function_modules_num_inputs = {}
        self.vocab = vocab
        for fn_str in vocab['program_token_to_idx']:
            num_inputs = iep.programs.get_num_inputs(fn_str)
            self.function_modules_num_inputs[fn_str] = num_inputs
            if fn_str == 'scene' or num_inputs == 1:
                mod = ResidualBlock(module_dim,
                                    with_residual=module_residual,
                                    with_batchnorm=module_batchnorm)
            elif num_inputs == 2:
                mod = ConcatBlock(module_dim,
                                  with_residual=module_residual,
                                  with_batchnorm=module_batchnorm)
            self.add_module(fn_str, mod)
            self.function_modules[fn_str] = mod

        self.save_module_outputs = False

    def add_module(self, name, module):
        self.modules[name] = module

    def expand_answer_vocab(self, answer_to_idx, std=0.01, init_b=-50):
        # TODO: This is really gross, dipping into private internals of Sequential
        final_linear_key = str(len(self.classifier._modules) - 1)
        final_linear = self.classifier._modules[final_linear_key]

        old_weight = final_linear.weight.data
        old_bias = final_linear.bias.data
        old_N, D = old_weight.shape
        new_N = 1 + max(answer_to_idx.values())

        new_weight = tf.random.normal(shape=[new_N, D])
        new_weight = tf.scalar_mul(std, new_weight)
        new_bias = tf.fill([new_N], init_b)
        new_weight[:old_N].copy_(old_weight)
        new_bias[:old_N].copy_(old_bias)

        final_linear.weight.data = new_weight
        final_linear.bias.data = new_bias

    def _forward_modules_json(self, feats, program):
        # def gen_hook(i, j):  # CHANGE
        #     def hook(grad):  # CHANGE
        #         self.all_module_grad_outputs[i][j] = grad.data.cpu().clone()  # CHANGE
        #
        #     return hook

        self.all_module_outputs = []
        self.all_module_grad_outputs = []
        # We can't easily handle minibatching of modules, so just do a loop
        N = tf.shape(feats)[0]
        final_module_outputs = []
        for i in range(N):
            if self.save_module_outputs:
                self.all_module_outputs.append([])
                self.all_module_grad_outputs.append([None] * len(program[i]))
            module_outputs = []
            for j, f in enumerate(program[i]):
                f_str = iep.programs.function_to_str(f)
                module = self.function_modules[f_str]
                if f_str == 'scene':
                    module_inputs = [feats[i:i + 1]]
                else:
                    module_inputs = [module_outputs[j] for j in f['inputs']]
                module_outputs.append(module(*module_inputs))
                if self.save_module_outputs:
                    self.all_module_outputs[-1].append(module_outputs[-1].read_value().numpy())  # CHANGE
                    #module_outputs[-1].register_hook(gen_hook(i, j))  # CHANGE
            final_module_outputs.append(module_outputs[-1])
        final_module_outputs = tf.concat(final_module_outputs, 0)
        return final_module_outputs

    def _forward_modules_ints_helper(self, feats, program, i, j):
        #print("fwd me feats ka shape : ", feats.shape)
        #print("prgm ki shape ", program.shape)
        used_fn_j = True
        if j < tf.shape(program)[1]:
            fn_idx = program.read_value().numpy()[i, j]
            fn_str = self.vocab['program_idx_to_token'][fn_idx]
        else:
            used_fn_j = False
            fn_str = 'scene'
        if fn_str == '<NULL>':
            used_fn_j = False
            fn_str = 'scene'
        elif fn_str == '<START>':
            used_fn_j = False
            return self._forward_modules_ints_helper(feats, program, i, j + 1)
        if used_fn_j:
            self.used_fns[i, j] = 1
        j += 1
        module = self.function_modules[fn_str]
        if fn_str == 'scene':
            #feats = tf.transpose(feats, perm=[0,2,3,1])
            module_inputs = [feats[i:i + 1]]
            #feats = tf.transpose(feats, perm=[0,3,1,2])
            #print("dekhte hai ", module_inputs[0].shape)
        else:
            num_inputs = self.function_modules_num_inputs[fn_str]
            module_inputs = []
            while len(module_inputs) < num_inputs:
                cur_input, j = self._forward_modules_ints_helper(feats, program, i, j)
                module_inputs.append(cur_input)
        #for i, item in enumerate(module_inputs):
        #     if True or item.shape != (1, 128, 14, 14):
        #         print("Shape of input #",i," : ", item.shape)
        module_output = module(*module_inputs)
        return module_output, j

    def _forward_modules_ints(self, feats, program):
        """
        feats: FloatTensor of shape (N, C, H, W) giving features for each image
        program: LongTensor of shape (N, L) giving a prefix-encoded program for
          each image.
        """
        N = tf.shape(feats)[0]
        final_module_outputs = []
        self.used_fns = tf.fill(tf.shape(program), 0)
        self.used_fns = self.used_fns.numpy()
        for i in range(N):
            cur_output, _ = self._forward_modules_ints_helper(feats, program, i, 0)
            #print("curr output shape : ", cur_output.shape)
            final_module_outputs.append(cur_output)
        self.used_fns = tf.convert_to_tensor(self.used_fns, dtype=tf.float32)
        #self.used_fns = self.used_fns.type_as(program.data).float()
        #print("lenght of final_module_outputs : ", len(final_module_outputs))
        #print("shape of final_module_outputs 0, 1, last: ", final_module_outputs[0].shape, final_module_outputs[1].shape, final_module_outputs[-1].shape)
        final_module_outputs = tf.concat(final_module_outputs, 0)
        #print("shape of final_module_outputs : ", final_module_outputs.shape)
        return final_module_outputs

    def __call__(self, x, program):
        N = tf.shape(x)[0]
        #assert N == len(program)

        feats = self.stem(x)
        #print("shape of feats :", feats.shape)
        feats = tf.transpose(feats, perm=[0, 3, 1, 2])
        #print("shape of feats :", feats.shape)
        #print(type(program), "is program type")
        #print("rank of program is : ", tf.rank(program))

        if type(program) is list or type(program) is tuple:
            final_module_outputs = self._forward_modules_json(feats, program)
        elif tf.rank(program) == 2:
            final_module_outputs = self._forward_modules_ints(feats, program)
        else:
            raise ValueError('Unrecognized program format')

        # After running modules for each input, concatenat the outputs from the
        # final module and run the classifier.
        out = self.classifier(final_module_outputs)
        return out
