Mmodule.cuda()
Mmodule.train()

criterion = nn.L1Loss()
optimizer = optim.Adam(Mmodule.parameters(), lr = 0.0001, betas=(0.5, 0.999))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 - max(0, step - num_epoches) / float(num_epoches + 1))

for epoch in range(num_epoches + num_epoches):
    iter_start_time = time.time()
    trainings = training_generator.next_batch()
    p1rep = trainings["p1"].cuda()
    p1rep = p1rep.squeeze(1)
    #print(p1rep.shape)
    c = trainings["cloth"].cuda()
    c = c.squeeze(1)
    y = trainings["label"].cuda()
    names = trainings["ID"]
    #print(c.shape)
    #print(y.shape)
    grid, theta = Mmodule(p1rep, c)
    warped_cloth = F.grid_sample(c, grid, padding_mode='border')

    loss = criterion(warped_cloth, y.squeeze(1))  
    optimizer.zero_grad()  
    loss.backward()
    optimizer.step()
    visuals = [ [c, warped_cloth, y.squeeze(1)]]

    if (epoch+1) % 100 == 0:
          board_add_images(board, 'combine', visuals, epoch+1)
          board.add_scalar('metric', loss.item(), epoch+1)
          t = time.time() - iter_start_time
          print('step: %8d, time: %.3f, loss: %4f' % (epoch+1, t, loss.item()), flush=True)
    
    if (epoch+1) % 1000 == 0:
            #q = q + 1
            save_images(warped_cloth, names, "gdrive/My Drive/Results/Pics")
            save_checkpoint(Mmodule, os.path.join("gdrive/My Drive", "M1", 'step_%06d.pth' % (epoch+1)))
