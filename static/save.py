    def get_epi_move_loss(self, uv, st, ori_e, ori_c, label=''):
        aug_points = uv.shape[0]
        outputs = {}

        progress = self.iter/max(self.epi_larger_remove_iter, self.iter)
        epi_range = self.init_epi_range*(1-progress) + self.min_epi_range*progress
        dis = torch.rand([aug_points,1], device=uv.device)-0.5
        dis = dis*2*self.max_moving_dis.max()*epi_range
        u_move, s_move = dis*ori_e.cos(), dis*ori_e.sin()

        us_move_uv = uv
        us_move_uv[:,:1] += u_move
        us_move_st = st
        us_move_st[:,:1] += s_move
        us_move_uv_embeds = self.embedder(us_move_uv)
        us_move_st_embeds = self.embedder(us_move_st)
        us_move_epi_dir, us_move_color_code, us_move_rgb = self.field([us_move_uv_embeds, us_move_st_embeds])
        outputs[f'{label}us_epi_loss'] = ((us_move_epi_dir-ori_e)**2)*self.epi_reg_weight
        outputs[f'{label}us_code_loss'] = ((us_move_color_code-ori_c)**2)*self.epi_reg_weight
        outputs[f'{label}us_proj_loss'] = torch.zeros_like(uv)
        if self.iter > self.epi_larger_remove_iter:
            smaller_mask = (us_move_epi_dir<ori_e)[:,0]
            outputs[f'{label}us_epi_loss'] = outputs[f'{label}us_epi_loss'][smaller_mask].mean()
            outputs[f'{label}us_code_loss'] = outputs[f'{label}us_code_loss'][smaller_mask].mean()
            # if (~smaller_mask).sum() > 0:
            #     outputs[f'{label}us_proj_loss'] = self.get_grid_proj_loss(
            #         us_move_uv[~smaller_mask], us_move_st[~smaller_mask], 
            #         us_move_epi_dir[~smaller_mask], us_move_color_code[~smaller_mask])

        vt_move_uv = uv
        vt_move_uv[:,1:] += u_move
        vt_move_st = st
        vt_move_st[:,1:] += s_move
        vt_move_uv_embeds = self.embedder(vt_move_uv)
        vt_move_st_embeds = self.embedder(vt_move_st)
        vt_move_epi_dir, vt_move_color_code, vt_move_rgb = self.field([vt_move_uv_embeds, vt_move_st_embeds])
        outputs[f'{label}vt_epi_loss'] = ((vt_move_epi_dir-ori_e)**2)*self.epi_reg_weight
        outputs[f'{label}vt_code_loss'] = ((vt_move_color_code-ori_c)**2)*self.epi_reg_weight
        outputs[f'{label}vt_proj_loss'] = torch.zeros_like(uv)
        if self.iter > self.epi_larger_remove_iter:
            smaller_mask = (vt_move_epi_dir<ori_e)[:,0]
            outputs[f'{label}vt_epi_loss'] = outputs[f'{label}vt_epi_loss'][smaller_mask].mean()
            outputs[f'{label}vt_code_loss'] = outputs[f'{label}vt_code_loss'][smaller_mask].mean()
            # if (~smaller_mask).sum() > 0:
            #     outputs[f'{label}vt_proj_loss'] = self.get_grid_proj_loss(
            #         vt_move_uv[~smaller_mask], vt_move_st[~smaller_mask], 
            #         vt_move_epi_dir[~smaller_mask], vt_move_color_code[~smaller_mask])

        return outputs
    
    
    
    
    
    
    
    # save epi for nb model
# uv_path = f"./data/out/all_uv_train.npy"
# st_path = f"./data/out/all_st_train.npy"
# color_path = f"./data/out/all_color_train.npy"
# all_uv = (dataset.all_uv[:,None,None]+torch.zeros_like(dataset.all_st)).cpu().numpy()
# all_st = dataset.all_st.cpu().numpy()
# all_color = dataset.all_color.cpu().numpy()
# np.save(uv_path, all_uv.reshape([-1,2])/dataset.scale)
# np.save(st_path, all_st.reshape([-1,2])/dataset.scale)
# np.save(color_path, all_color.reshape([-1,3]))
# val_uv_path = f"./data/out/all_uv_render.npy"
# val_st_path = f"./data/out/all_st_render.npy"
# min_u, min_v = dataset.all_uv.min(0).values
# max_u, max_v = dataset.all_uv.max(0).values
# u_seq = torch.cat([
#     torch.linspace(min_u, max_u, 10),
#     torch.linspace(max_u, min_u, 10),
#     torch.linspace(min_u, min_u, 10),
# ], dim=0)
# v_seq = torch.cat([
#     torch.linspace(min_v, max_v, 10),
#     torch.linspace(max_v, max_v, 10),
#     torch.linspace(max_v, min_v, 10),
# ], dim=0)
# val_all_uv = []
# val_all_st = []
# for i, (u, v) in enumerate(zip(u_seq, v_seq)):
#     uv = torch.tensor([u,v]).to(dataset.st_base.device).expand_as(dataset.st_base)
#     val_all_uv.append(uv.cpu().numpy().reshape([-1,2])/dataset.scale)
#     st = (dataset.st_base + uv)/dataset.scale
#     val_all_st.append(st.cpu().numpy().reshape([-1,2]))
# np.save(val_uv_path, np.concatenate(val_all_uv, 0).reshape([-1,2]))
# np.save(val_st_path, np.concatenate(val_all_st, 0).reshape([-1,2]))
# import pdb;pdb.set_trace()

    
    
    
    # nb_model
    
        all_uv = []
        all_st = []
        num_img=self.imgs.shape[0]
        for pose in self.poses:
            rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose[:3,:4])
            rays_ori, rays_dir = rays_ori.reshape([-1,3]), rays_dir.reshape([-1,3])
            all_uv.append(get_line_plane_collision(rays_ori, rays_dir, uv_plane))
            all_st.append(get_line_plane_collision(rays_ori, rays_dir, st_plane))
        all_uv=torch.cat(all_uv,dim=0)
        all_st=torch.cat(all_st,dim=0)
        all_color=self.imgs.reshape([num_img,-1,3]).reshape(-1,3)
        np.save(f'./data/out/all_uv_{self.split}.npy', all_uv.numpy())
        np.save(f'./data/out/all_st_{self.split}.npy', all_st.numpy())
        np.save(f'./data/out/all_color_{self.split}.npy', all_color.numpy())
    
        all_uv = []
        all_st = []    
        for pose in self.render_poses:
            pose = torch.tensor(pose)
            rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose[:3,:4])
            rays_ori, rays_dir = rays_ori.reshape([-1,3]), rays_dir.reshape([-1,3])
            all_uv.append(get_line_plane_collision(rays_ori, rays_dir, uv_plane))
            all_st.append(get_line_plane_collision(rays_ori, rays_dir, st_plane))
        all_uv=torch.cat(all_uv,dim=0)
        all_st=torch.cat(all_st,dim=0)
        np.save(f'./data/out/all_uv_render.npy', all_uv.numpy())
        np.save(f'./data/out/all_st_render.npy', all_st.numpy())
        np.save(f'./data/out/all_color_render.npy', all_color.numpy())



        # save all points to obj
        def save(pose_i):
            pose = self.poses[pose_i, :3,:4]
            rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)
            point = []
            all_num = 10
            rays_ori = rays_ori.cpu().numpy()[::5,::5,:].reshape([-1,3])
            rays_dir = rays_dir.cpu().numpy()[::5,::5,:].reshape([-1,3])
            for i in range(all_num):
                point.append(rays_ori+rays_dir*(i+1)/all_num*self.far)
            point = np.stack(point).reshape([-1,3])

            p_point = []
            x0 = -0.03301038
            y0 = -0.05038188
            all_num = 100
            for i in range(all_num):
                for j in range(all_num):
                    x = x0+(i*1./all_num-0.5)*1
                    y = y0+(j*1./all_num-0.5)*1
                    z = -0.055935*x - 0.012899*y - 1.18754
                    p_point.append([x,y,z])
            p_point = np.stack(p_point).reshape([-1,3])

            with open(f'./data/out/point{pose_i}.obj', 'w') as f:
                for p in point:
                    f.write(f'v {p[0]} {p[1]} {p[2]} 1.0 0.0 0.0\n')
                for p in p_point:
                    f.write(f'v {p[0]} {p[1]} {p[2]} 0.0 1.0 0.0\n')
        for i in range(len(self.poses)):
            save(i)
        import pdb;pdb.set_trace()



