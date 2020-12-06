class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs):
        #self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_one_channel = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])
        self.main_directory = "data/train/"
        self.image_directory = "image/"
        self.parse_directory = "image-parse/"
        self.cloth_directory = "cloth/"
        self.pose_directory = "pose/"

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        image = self.main_directory + self.image_directory + ID
        pose = self.main_directory + self.pose_directory + ID
        parse = self.main_directory + self.parse_directory + ID
        cloth = self.main_directory + self.cloth_directory + ID
        
        # Contsructing the samples
        parsed_image = parse.replace('.jpg', '.png')
        cloth = cloth.replace('_0', '_1')
        pose = pose.replace('.jpg', '_keypoints.json')
        
        
        im = Image.open(image)
        cl = Image.open(cloth)
        pim = Image.open(parsed_image)
        im_tensor = self.transform(im)
        
        # Images to npArray
        image_array = np.array(im)
        pim_array = np.array(pim)
        
        #making the pose map with 18 points
        with open(pose, 'r') as f:
            jf = json.load(f)
            jf = jf['people'][0]['pose_keypoints']
            jf = np.array(jf)
            jf = jf.reshape((-1, 3))
        
        number_of_points = jf.shape[0]
        r = 3
        pose_map = torch.zeros(number_of_points, 256, 192)
        for i in range(0, 17):
            one_map = Image.new('RGB', (192, 256))
            draw = ImageDraw.Draw(one_map)
            pointx = jf[i, 0]
            pointy = jf[i, 1]
            if pointx > 1 and pointy > 1:
                    draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                    #pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                    h = one_map
                    one_map = self.transform(one_map)
                    pose_map[i] = one_map[0]
        
        #open the parsing in one channel
        immm = Image.open(parsed_image).convert('L')
        x = self.transform_one_channel(immm)
        one_channel_parsed_data = x
        
        #contsructing p1
        p1 = torch.zeros(1, 19, 256, 192)
        reshaped_pose_map = pose_map.reshape((1, 18, 256, 192))
        reshaped_one_channel_parsed_data = one_channel_parsed_data.reshape((1, 1, 256, 192))
        concating_p1_parts = torch.cat([reshaped_pose_map[0], reshaped_one_channel_parsed_data[0]], 0)  
        p1[0] = concating_p1_parts
        p1 = p1.squeeze(0)
        #print(p1.shape)
        
        # Target cloth
        cl_transform = self.transform(cl)
        cl_transform = cl_transform.unsqueeze(0)
        #print(cl_transform.shape)
        
         
        # finding the head
        parse_head = (pim_array == 1).astype(np.float32) + (pim_array == 2).astype(np.float32) + (pim_array == 4).astype(np.float32) + (pim_array == 13).astype(np.float32)
        
        #finding the cloth
        cloth_parsed = (pim_array == 5).astype(np.float32) + (pim_array == 6).astype(np.float32) + (pim_array == 7).astype(np.float32)
        #cloth_parsed = torch.from_numpy(cloth_parsed)
        
        #finding the shape
        parse_shape = (pim_array != 0).astype(np.float32)
        
        #finding the arms
        parse_arm = (pim_array == 14).astype(np.float32) + (pim_array == 15).astype(np.float32)
        
        #tensors
        parse_shape = transform_one_channel(parse_shape)
        parse_shape = parse_shape.reshape((1, 1, 256, 192))
        parse_head = transform_one_channel(parse_head)
        parse_head = parse_head.reshape((1, 1, 256, 192))
        parse_arm = transform_one_channel(parse_arm)
        parse_arm = parse_arm.reshape((1, 1, 256, 192))

        parse_cloth = transform_one_channel(cloth_parsed)
        reshaped_cloth_parsed = cloth_parsed.reshape((1, 1, 256, 192))
        
        im_h = im_tensor * parse_head + ( 1 - parse_head )
        
        #p2
        p2 = torch.zeros(1, 22, 256, 192)
        concating_p2_parts = torch.cat([reshaped_pose_map[0], parse_shape[0], parse_arm[0], parse_head[0], reshaped_cloth_parsed[0]], 0)  
        p2[0] = concating_p2_parts
        
        #p3
        p3 = torch.zeros(1, 23, 256, 192)
        concating_p3_parts = torch.cat([reshaped_pose_map[0], one_channel_parsed_data, parse_arm[0], im_h[0]], 0) 
        p3[0] = concating_p3_parts
        
        
        # it is our y *.*
        im_c = im_tensor * cloth_parsed + ( 1 - cloth_parsed )
        im_c = im_c.unsqueeze(0)
        
        res = {
            'label': im_c,
            'p1': p1,
            'cloth': cl_transform,
            'ID' : ID,
            'p2': p2,
            'p3': p3,
            'segmentation map': one_channel_parsed_data,
            
        }
        
        return res
    
    def __len__(self):
        return len(self.list_IDs)
    
    
    
    
class CPDataLoader(object):
    def __init__(self, training_set, params):
        #training_set = Dataset(dataset)
        self.data_loader = torch.utils.data.DataLoader(training_set, **params)
        self.dataset = training_set
        self.data_iter = self.data_loader.__iter__()
        
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
