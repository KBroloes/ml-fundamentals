from tensorflow import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

# Docs: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
class PlotCallback(keras.callbacks.Callback):
    
    def __init__(self,max_epochs,print_every=5, window=100):
        self.max_epochs = max_epochs
        self.print_every = print_every
        self.window = window
        
    def plot(self, epoch):
        #score = model.evaluate(X_test, y_test, verbose=0)
        clear_output(wait=True)
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        plt.suptitle(f"Epoch {epoch} out of {self.max_epochs}")
        
        # Accuracy plot
        ax[0].plot(self.accuracy[-self.window:])
        ax[0].plot(self.val_accuracy[-self.window:])

        ax[0].set_title(f'Model Accuracy [Test: {self.val_accuracy[-1]*100:.2f}%]')
        ax[0].set_ylabel('accuracy')
        ax[0].set_xlabel('epoch')
        ax[0].legend(['train', 'validation'], loc='upper left')
        
        # Loss plot
        ax[1].plot(self.loss[-self.window:])
        ax[1].plot(self.val_loss[-self.window:])
        ax[1].set_title(f'Model loss [Test: {self.val_loss[-1]:.2f}]')
        ax[1].set_ylabel('loss')
        ax[1].set_xlabel('epoch')
        ax[1].legend(['train', 'validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()


    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    # This function is called at the end of training
    def on_train_end(self, logs={}):
        self.plot(self.max_epochs)
        
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_accuracy.append(logs.get('val_accuracy'))

        if epoch % self.print_every == 0:
            self.plot(epoch)

def plot_confusion_matrix(cm, class_names=None):
    classes = len(cm)

    # Calculate accuracy for the plot title
    accuracy = 100*cm.diagonal().sum()/cm.sum()
    error_rate = 100-accuracy


    # Plot the confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='None')
    plt.colorbar(format='%.2f')
    plt.xticks(range(classes), labels=class_names)
    plt.yticks(range(classes), labels=class_names)

    plt.xlabel('Predicted class')
    plt.ylabel('Actual class')
    plt.title('Confusion matrix (Accuracy: {:.2f}%, Error Rate: {:.2f}%)'.format(accuracy, error_rate))
    plt.plot()

    # Fill the plot with the corresponding values
    threshold = cm.max() / 2.0 # Threshold for using white or black text
    for x in range(classes):
        for y in range(classes):
            plt.text(y, x, format(cm[x, y], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[x, y] > threshold else "black") # Some pretty printing to make it more legible

def build_graph(model, width=40, height=20, max_units=10, merge_dropout=True):
    conf = model.get_config()
    name = conf['name']
    layers = conf['layers']
    dot = f'''
    digraph G {{

        graph[ fontname = "Roboto",
               fontsize = 16,
               label = "",
               size = "{width},{height}" ];
        rankdir = LR;
        splines=false;
        edge[style=invis];
        ranksep=1.4;

    '''
    if max_units < 2:
        raise(Exception("Max units must be 2 or more"))
    
    nodes = []
    dropout = []
    for i, layer in enumerate(layers):
        class_name = layer['class_name']
        l_conf = layer['config']
        name = l_conf['name']
        
        if class_name == 'InputLayer':
            shape = l_conf['batch_input_shape']
            units = shape[1]
            color = '#00F082'
            prefix = 'input'

            l_label = f"{name} ({prefix})"
        elif class_name == 'Dropout':
            if merge_dropout:
                continue
                
            rate = l_conf['rate']
            prefix = 'dropout'
            units = 1
            color = "#FF1EC8"
            
            l_label = f"{name} ({prefix}: {rate})"
        else:
            units = l_conf['units']
            act = l_conf['activation']
            add_label = ""
            
            if i == len(layers)-1:
                color = '#FF640A'
                prefix = 'output'
            else:
                color = '#0AB4FA'
                prefix = 'hidden'
                
                if merge_dropout:
                    # Only works when we're not in an output layer yet
                    next_layer = layers[i+1]
                    if next_layer['class_name'] == 'Dropout':
                        color = "#FF1EC8"
                        rate = next_layer['config']['rate']
                        add_label = f"[Dropout: {rate}]"
            
            l_label = f"{name} ({prefix} - {act})" + add_label
            
        
        node_ids = range(units)
        if units > max_units:
            mid = int(max_units/2)
            node_ids = list(np.concatenate([np.arange(mid), ['truncated'], np.arange(units-mid, units)]))
        
        layer_nodes = [f"{prefix}_{i}_u{j}" for j in node_ids]
        
        nodes.append(layer_nodes)
        nodelist = ";".join([f"{prefix}_{i}_u{j} [label=<{prefix}_{j}>]" for j in node_ids])
        ranklist = "->".join([f"{prefix}_{i}_u{j}" for j in node_ids])

        layer_def = f'{name} [shape=plaintext, label="{l_label} ({units} units)"];'
        layer_obj = f'''
        {{
            node [shape=circle, color="{color}", style=filled, fillcolor="{color}"];
            {nodelist};
            {{
                rank=same
                {ranklist};
            }}
            
        }}
        {layer_def}
        {name}->{prefix}_{i}_u0
        {{rank=same; {name}->{prefix}_{i}_u0;}}
        '''
        
        dot += layer_obj
        
    edges = """
        edge[style=solid, tailport=e, headport=w];
        
    """
    for i, layer_nodes in enumerate(nodes):
        if i == len(nodes)-1:
            break
        edge = f"{{{';'.join(layer_nodes)}}} -> {{{';'.join(nodes[i+1])}}};\n"
        edges += edge
    
    dot += edges
    return dot + "}"

def render_network(model, width=1000, height=600, retina=True, max_units=10, merge_dropout=True):
    from graphviz import Source
    from IPython.display import Image, display
    from os import remove
    graph = build_graph(model, width=width, height=height, max_units=max_units, merge_dropout=merge_dropout)
    dot = Source(graph, format='png')
    display(Image(dot.render('temp'), retina=retina))
    remove('temp.png')