class FeatureExtraction(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(nc, 64, stride = 2, kernel_size = 3)
        self.conv2 = nn.Conv2d(64, 128, stride = 2, kernel_size = 3)
        self.conv3 = nn.Conv2d(128, 256, stride = 2, kernel_size = 3)
        self.conv4 = nn.Conv2d(256, 512, stride = 2, kernel_size = 3)
        self.conv5 = nn.Conv2d(512, 512, stride = 1, kernel_size = 3)
        self.conv6 = nn.Conv2d(512, 512, stride = 1, kernel_size = 3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x



class M1(nn.Module):
    def __init__(self):
        super().__init__()
        self.featuresA = FeatureExtraction(19)
        self.featuresB = FeatureExtraction(3)
        self.NLA = NLBlockND(in_channels = 512, dimension = 2)
        self.NLB = NLBlockND(in_channels = 512, dimension = 2)
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=77, output_dim=2*3**2, use_cuda=True)
        self.TPS = TpsGridGen(256, 192, use_cuda=True, grid_size=3)
        
    def forward(self, inputA, inputB):
        featureA = self.featuresA(inputA)
        featureB = self.featuresB(inputB)
        #print(featureA.shape)
        featuresAwithNL = self.NLA(featureA)
        featuresBwithNL = self.NLB(featureB)
        featerscorrelation = self.correlation(featureA, featureB)
        #print(featerscorrelation.shape)
        theta = self.regression(featerscorrelation)
        grid = self.TPS(theta)
        return grid, theta
        




class Attention(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nc, 1, stride = 2, kernel_size = 3)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(1, nc, stride = 2, kernel_size = 3)
        self.BN = nn.BatchNorm2d(nc)
        
    def forward(self, input1, input2):
        x = input1 + input2
        x = self.conv1(self.relu1(x))
        x = self.sigmoid1(x)
        x = input1 * x
        x = self.conv2(x)
        x = self.BN(x)
        return x
         
         
 #inputs = p2, cloth normal
#output = segmap
class SegmentationMapGAN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.firstconvolution =  nn.Conv2d(nc, 64, stride = 2, kernel_size = 4)
        
        self.layer1 = nn.Sequential()
        self.layer1.add_module("LR1", nn.LeakyReLU())
        self.layer1.add_module("conv2", nn.Conv2d(64, 128, stride = 2, kernel_size = 4))
        self.layer1.add_module("BN1", nn.BatchNorm2d(128))
        
        self.layer2 = nn.Sequential()
        self.layer2.add_module("LR2", nn.LeakyReLU())
        self.layer2.add_module("conv3", nn.Conv2d(128, 256, stride = 2, kernel_size = 4))
        self.layer2.add_module("BN2", nn.BatchNorm2d(256))
        
        self.layer3 = nn.Sequential()
        self.layer3.add_module("LR3", nn.LeakyReLU())
        self.layer3.add_module("conv4", nn.Conv2d(256, 512, stride = 2, kernel_size = 4))
        self.layer3.add_module("BN3", nn.BatchNorm2d(512))
        
        self.NL1 = NLBlockND(in_channels = 512, dimension = 2)
        self.NL2 = NLBlockND(in_channels = 256, dimension = 2)
        self.NL3 = NLBlockND(in_channels = 128, dimension = 2)
        
        self.layer4 = nn.Sequential()
        self.layer4.add_module("R1", nn.ReLU())
        self.layer4.add_module("deconv1", nn.ConvTranspose2d(1024, 512, stride = 2, kernel_size = 4))
        self.layer4.add_module("BN4", nn.BatchNorm2d(1024))
        self.layer4.add_module("Drop1", nn.Dropout(0.2))
        
        self.layer5 = nn.Sequential()
        self.layer5.add_module("R2", nn.ReLU())
        self.layer5.add_module("deconv2", nn.ConvTranspose2d(512, 256, stride = 2, kernel_size = 4))
        self.layer5.add_module("BN5", nn.BatchNorm2d(256))
        
        self.layer6 = nn.Sequential()
        self.layer6.add_module("R3", nn.ReLU())
        self.layer6.add_module("deconv3", nn.ConvTranspose2d(256, 128, stride = 2, kernel_size = 4))
        self.layer6.add_module("BN6", nn.BatchNorm2d(128))
        
        self.layer7 = nn.Sequential()
        self.layer7.add_module("R4", nn.ReLU())
        self.layer7.add_module("deconv4", nn.ConvTranspose2d(128, 1, stride = 2, kernel_size = 4))
        self.layer7.add_module("Th1", nn.Tanh())
        
        #cloth_features
        
        self.layer8 = nn.Sequential()
        self.layer8.add_module("conv8", nn.Conv2d(3, 32, stride = 2, kernel_size = 3))
        self.layer8.add_module("BN8", nn.BatchNorm2d(32))
        self.layer8.add_module("R8", nn.ReLU())
        
        self.layer9 = nn.Sequential()
        self.layer9.add_module("conv9", nn.Conv2d(32, 64, stride = 2, kernel_size = 3))
        self.layer9.add_module("BN9", nn.BatchNorm2d(64))
        self.layer9.add_module("R9", nn.ReLU())
        
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,512,2,2,start_with_relu=True,grow_first=True)
        
        self.attention1 = Attention(256)
        self.attention2 = Attention(128)
        self.attention3 = Attention(64)
        
        
    def forward(self, inputA, inputB):
        x_l1 = self.firstconvolution(inputA)
        x_l2 = self.layer1(x_l1)
        x_l3 = self.layer2(x_l2)
        x_l4 = self.layer3(x_l3)
        
        y = self.layer8(inputB)
        y = self.layer9(y)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block33(y)
        
        x = torch.cat((x_l3, y), 0)
        x_l4 = self.layer4(x)
        x_l4_a = self.attention1(x_l3, x_l4)
        x = torch.cat((x_l4, x_l4_a), 0)
        x = self.NL1(x)
        
        x_l5 = self.layer5(x)
        x_l5_a = self.attention1(x_l2, x_l5)
        x = torch.cat((x_l5, x_l5_a), 0)
        x = self.NL2(x)
        
        x_l6 = self.layer6(x)
        x_l6_a = self.attention1(x_l1, x_l6)
        x = torch.cat((x_l6, x_l6_a), 0)
        x = self.NL3(x)
        
        x = self.layer7(x)
        
        return x
        
