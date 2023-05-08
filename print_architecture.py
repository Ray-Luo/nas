import os
import torch
from dynamic_model import UNetMobileNetv3

generator = UNetMobileNetv3(512)
generator.load_state_dict(torch.load('./my_model.pth'))
generator.eval()

with torch.no_grad():
    print("repeat_mask_1: ", torch.sum(torch.relu(generator.repeat_mask_1)).item())
    print("out_channel_mask_1: ", torch.sum(torch.relu(generator.out_channel_mask_1)).item())
    for i in range(len(generator.db1)):
        cur = generator.db1[i]
        print("irb_bottleneck1 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("irb_bottleneck1 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("irb_bottleneck1 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    print("repeat_mask_2: ", torch.sum(torch.relu(generator.repeat_mask_2)).item())
    print("out_channel_mask_2: ", torch.sum(torch.relu(generator.out_channel_mask_2)).item())
    for i in range(len(generator.db2)):
        cur = generator.db2[i]
        print("irb_bottleneck2 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("irb_bottleneck2 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("irb_bottleneck2 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    print("repeat_mask_3: ", torch.sum(torch.relu(generator.repeat_mask_3)).item())
    print("out_channel_mask_3: ", torch.sum(torch.relu(generator.out_channel_mask_3)).item())
    for i in range(len(generator.db3)):
        cur = generator.db3[i]
        print("irb_bottleneck3 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("irb_bottleneck3 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("irb_bottleneck3 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    print("repeat_mask_4: ", torch.sum(torch.relu(generator.repeat_mask_4)).item())
    print("out_channel_mask_4: ", torch.sum(torch.relu(generator.out_channel_mask_4)).item())
    for i in range(len(generator.db4)):
        cur = generator.db4[i]
        print("irb_bottleneck4 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("irb_bottleneck4 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("irb_bottleneck4 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    print("repeat_mask_5: ", torch.sum(torch.relu(generator.repeat_mask_5)).item())
    print("out_channel_mask_5: ", torch.sum(torch.relu(generator.out_channel_mask_5)).item())
    for i in range(len(generator.db5)):
        cur = generator.db5[i]
        print("irb_bottleneck5 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("irb_bottleneck5 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("irb_bottleneck5 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    print("repeat_mask_6: ", torch.sum(torch.relu(generator.repeat_mask_6)).item())
    print("out_channel_mask_6: ", torch.sum(torch.relu(generator.out_channel_mask_6)).item())
    for i in range(len(generator.db6)):
        cur = generator.db6[i]
        print("irb_bottleneck6 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("irb_bottleneck6 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("irb_bottleneck6 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    print("out_channel_mask_7: ", torch.sum(torch.relu(generator.out_channel_mask_7)).item())
    for i in range(len(generator.db7)):
        cur = generator.db7[i]
        print("irb_bottleneck7 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("irb_bottleneck7 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("irb_bottleneck7 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    for i in range(len(generator.ub1)):
        cur = generator.ub1[i]
        print("D_irb1 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("D_irb1 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("D_irb1 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    for i in range(len(generator.ub2)):
        cur = generator.ub2[i]
        print("D_irb2 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("D_irb2 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("D_irb2 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    for i in range(len(generator.ub3)):
        cur = generator.ub3[i]
        print("D_irb3 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("D_irb3 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("D_irb3 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    for i in range(len(generator.ub4)):
        cur = generator.ub4[i]
        print("D_irb4 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("D_irb4 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("D_irb4 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    for i in range(len(generator.ub5)):
        cur = generator.ub5[i]
        print("D_irb5 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("D_irb5 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("D_irb5 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")

    for i in range(len(generator.ub6)):
        cur = generator.ub6[i]
        print("D_irb6 {} expansion_mask: ".format(i), torch.sum(torch.relu(cur.expansion_mask)).item())
        print("D_irb6 {} act_mask: ".format(i), torch.relu(cur.act_mask).item())
        print("D_irb6 {} se_mask: ".format(i), torch.relu(cur.se_mask).item())
    print("\n")
