import torch.nn as nn
import torch
import numpy as np
 
def weight_init(m):
	if isinstance(m, nn.Conv2d):
		size = m.weight.size()
		fan_out = size[0] # number of rows
		fan_in = size[1] # number of columns
		variance_w = np.sqrt(2.0/(fan_in + fan_out))
		m.weight.data.normal_(0.0, variance_w)

        
class general_net(nn.Module):
    def __init__(self, layers_sizes,in_sizer, pooling_layers, number_of_classes, kernel_sizes, image_shape, residuals, network_type, dropeout, dropeout_prob, batchnorm, activation):
        super(general_net, self).__init__()      
        
        self.convolution=nn.ModuleList();
        self.activation=nn.ModuleList();
        self.pool=nn.ModuleList();
        self.input_saturator=nn.Threshold(1, 1, inplace=False)
        if residuals==True:
                self.residual_shaping=nn.ModuleList();
        if batchnorm==True:
                self.bn=nn.ModuleList();
        if dropeout==True:
                self.do=nn.ModuleList();
                
        
                
        for layer_number in range(len(layers_sizes)):
            if layer_number==0:
                self.convolution.append(nn.Conv2d(in_sizer, layers_sizes[layer_number], kernel_size=kernel_sizes[layer_number], stride=1, padding=((kernel_sizes[layer_number]-1)//2), bias=True));
            else:
                self.convolution.append(nn.Conv2d(layers_sizes[layer_number-1], layers_sizes[layer_number], kernel_size=kernel_sizes[layer_number], stride=1, padding=((kernel_sizes[layer_number]-1)//2), bias=True));

                
            self.convolution[layer_number].apply(weight_init)
            if activation=='ELU':
                self.activation.append(nn.ELU());
                self.activation_fully1=nn.ELU();
                self.activation_fully2=nn.ELU();
            else:
                self.activation.append(nn.ReLU());
                self.activation_fully1=nn.ReLU();
                self.activation_fully2=nn.ReLU();

            
            if residuals==True:
                if layer_number==0:
                    self.residual_shaping.append(nn.Conv2d(in_sizer, layers_sizes[layer_number], kernel_size=1, padding=0, bias=False))
                else:
                    self.residual_shaping.append(nn.Conv2d(layers_sizes[layer_number-1], layers_sizes[layer_number], kernel_size=1, padding=0, bias=False))
            if batchnorm==True:
                self.bn.append(nn.BatchNorm2d(layers_sizes[layer_number]));
                
            if dropeout==True:
                self.do.append(nn.Dropout2d(p=dropeout_prob))
                self.drop_fully=nn.Dropout2d(p=dropeout_prob*2)
                self.drop_fully1=nn.Dropout2d(p=dropeout_prob*2)
                self.drop_fully2=nn.Dropout2d(p=dropeout_prob*2)
            
            
            self.pool.append(nn.MaxPool2d(2, stride=2))
            
        image_shape_reduced=image_shape.copy()
        for i in range(len(pooling_layers)):
            image_shape_reduced[0]=image_shape_reduced[0]//2;
            image_shape_reduced[1]=image_shape_reduced[1]//2;

            
        if network_type=='fully_connected':
                
            self.fully_connected = nn.Linear(image_shape_reduced[0]*image_shape_reduced[1]*layers_sizes[-1]+1, 128)
            
            self.fully_connected1 = nn.Linear(128, 256)
            self.fully_connected2 = nn.Linear(256, number_of_classes)

        else:
            self.averaging=nn.AvgPool2d(kernel_size=image_shape_reduced[0]);
            self.fully_connected = nn.Linear(2, number_of_classes)
            
        self.softmax=nn.Softmax();
        self.sigmoid=nn.Sigmoid();
        
        self.residuals=residuals;
        self.layers_sizes=layers_sizes;
        self.batchnorm=batchnorm;
        self.pooling_layers=pooling_layers;
        self.dropeout=dropeout;
        self.network_type=network_type;
        self.in_sizer=in_sizer
        
    def forward(self, x, y):
        x=x.float();
      #  x=x-self.input_saturator(x)+1;
       # print(x)
        l_out=[];
         
        ## Passing through layers
        for layer_number in range(len(self.layers_sizes)):
            
            if self.residuals==True:
                residual = x;
             
            x=self.convolution[layer_number](x);
            x=self.activation[layer_number](x);
            if self.residuals==True:
                x=x+self.residual_shaping[layer_number](residual);
            
            if layer_number in self.pooling_layers:
                x=self.pool[layer_number](x)
            
            if self.batchnorm==True:
                x=self.bn[layer_number](x)

            if self.dropeout==True:
                x=self.do[layer_number](x)
                
            l_out.append(x);
            
        if self.network_type=='fully_connected':
            x = x.view(x.size(0), -1);
            x=torch.cat((x, y),1)
            x=self.drop_fully(x)
            x=self.fully_connected(x);
            x=self.activation_fully1(x);
            x=self.drop_fully1(x)
            x=self.fully_connected1(x);
            x=self.activation_fully2(x);
            x=self.drop_fully2(x)
            x=self.fully_connected2(x);
        else:
             
            x=self.averaging(x)
            x = x.view(x.size(0), -1)
 
            x = torch.cat((x, x),1)
            
            x=self.fully_connected(x);
            
        out = self.sigmoid(x)
 
        out_probas=self.softmax(x);
        
        return out


def neural_network(image_shape,in_sizer=3, layers_sizes=[10, 10, 10, 10], pooling_layers=[1,3], kernel_sizes=[3, 3, 3, 3], number_of_classes=2, residuals=True, network_type='fully_connected', dropeout=False, dropeout_prob=0.25, batchnorm=True,activation='ELU', **kwargs):
    model = general_net(layers_sizes,in_sizer, pooling_layers, number_of_classes, kernel_sizes, image_shape, residuals, network_type, dropeout, dropeout_prob, batchnorm, activation, **kwargs)
    return model
    
    
#### Network paralmeters definition
image_shape=[75,75];   
layers_sizes=[64,128,128,64,32,32,16,16,8,8]
kernel_sizes=[5,5,5,5,3,3,3,3,3,3]
pooling_layers=[1,4,6]
in_sizer=3;
residuals=True;
network_type='fully_connected'
dropeout=True;
batchnorm=False;
activation='ELU';
dropeout_prob=0.15;
network=neural_network(image_shape=image_shape, in_sizer=in_sizer,layers_sizes=layers_sizes, pooling_layers=pooling_layers, kernel_sizes=kernel_sizes, number_of_classes=1, residuals=residuals, network_type=network_type, dropeout=dropeout, dropeout_prob=dropeout_prob, batchnorm=batchnorm, activation=activation).cuda().float();
