from util import *


class SubLog:
    def __init__(self, process_id=None, log_path=None):
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.episode_metrics_result = {}
        self.episode_metrics_result['eff'] = []
        self.episode_metrics_result['fairness'] = []
        self.episode_metrics_result['dcr'] = []
        self.episode_metrics_result['hit'] = []
        self.episode_metrics_result['ec'] = []
        self.episode_metrics_result['ecr'] = []
        self.episode_metrics_result['cor'] = []

        self.root_log_path = log_path
        self.log_path = os.path.join(log_path, 'process_' + str(process_id))

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # for draw
        self.color_list = ['C' + str(color_id) for color_id in range(10)]
        self.marker_list = ['o', '.', 'v', '^', '<', '>', '1', '2', '3', '4']

    def record_sub_rollout_dict(self, sub_rollout_manager):
        sub_rollout_dict_path = os.path.join(self.log_path, 'sub_rollout_dict.npy')
        np.save(sub_rollout_dict_path, sub_rollout_manager.sub_rollout_dict)

    def gen_metrics_result(self, iter_id, env):
        # fairness
        fairness = 0.0
        final_poi_visit_time = np.clip(env.episode_log_info_dict['final_poi_visit_time'][-1], 0, 2)
        square_of_sum = np.square(np.sum(final_poi_visit_time))
        sum_of_square = np.sum(np.square(final_poi_visit_time))
        if sum_of_square > 1e-5:
            fairness = square_of_sum / sum_of_square / final_poi_visit_time.shape[0]
        self.episode_metrics_result['fairness'].append(fairness)

        # data_collection_ratio (dcr)
        dcr = np.sum(env.poi_init_value_array - env.poi_cur_value_array) / np.sum(env.poi_init_value_array)
        self.episode_metrics_result['dcr'].append(dcr)

        # hit
        hit = env.final_total_hit
        self.episode_metrics_result['hit'].append(hit)

        # energy_consumption (ec)
        ec = env.final_energy_consumption
        self.episode_metrics_result['ec'].append(ec)

        # energy_consumption_ratio (ecr)
        ecr = ec / env.ec_upper_bound
        self.episode_metrics_result['ecr'].append(ecr)

        # UGV_UAV_cooperation_ratio (cor)
        if env.final_total_relax_time > 0:
            cor = env.final_total_eff_relax_time / env.final_total_relax_time
        else:
            cor = 0
        self.episode_metrics_result['cor'].append(cor)

        # eff
        eff = 0.0
        if ecr > min_value:
            eff = fairness * dcr * cor / ecr
        self.episode_metrics_result['eff'].append(eff)

    def record_metrics_result(self):
        np.save(self.log_path + '/episode_metrics_result.npy', self.episode_metrics_result)

    ######################## trace_plot ############################
    def record_trace(self, iter_id, env, sub_rollout_manager, mode='train', subp_id=None):
        mpl.style.use('default')
        # Fig = plt.figure(figsize=(10, 10))
        Fig, ax = plt.subplots(figsize=(10, 10))
        plt.xlim(xmin=0, xmax=self.dataset_conf['coordx_max'])
        plt.ylim(ymin=0, ymax=self.dataset_conf['coordy_max'])
        # plt.grid(True, linestyle='-.', color='r')

        cm = plt.cm.get_cmap('RdYlBu_r')

        # draw trace
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            UGV_UAVs_Group = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id]
            # draw UGV trace
            ugv_id = UGV_UAVs_Group_id
            if self.method_conf['ugv_trace_type'] == 'roads_net':
                final_passed_road_node_id_list = \
                UGV_UAVs_Group.ugv.episode_log_info_dict['final_passed_road_node_id_list'][
                    -1]
                road_pos_list = UGV_UAVs_Group.ugv.episode_log_info_dict['final_road_pos']
                for item_id, passed_road_node_id_list in enumerate(final_passed_road_node_id_list):
                    ugv_coordx_list = []
                    ugv_coordy_list = []
                    ugv_coordx_list.append(env.road_pos2pos(road_pos_list[item_id])[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos_list[item_id])[1])
                    for passed_road_node_id in passed_road_node_id_list:
                        ugv_coordx_list.append(env.roads_net_dict[passed_road_node_id]['coordxy'][0])
                        ugv_coordy_list.append(env.roads_net_dict[passed_road_node_id]['coordxy'][1])
                    ugv_coordx_list.append(env.road_pos2pos(road_pos_list[item_id + 1])[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos_list[item_id + 1])[1])
                    plt.plot(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id], linewidth=2,
                             alpha=0.5)

                ugv_coordx_list = []
                ugv_coordy_list = []
                for road_pos in road_pos_list:
                    ugv_coordx_list.append(env.road_pos2pos(road_pos)[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos)[1])
                plt.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id],
                            marker=self.marker_list[0], s=10)

                status_list = UGV_UAVs_Group.episode_log_info_dict['status']
                ugv_stop_step_phase_list = []
                ugv_stop_step_phase_list.append([])
                for status_id, status in enumerate(status_list):
                    if status_id == 0:
                        if status == 2:
                            ugv_stop_step_phase_list[-1].append(status_id)
                    else:
                        if status_list[status_id - 1] == 3 and status_list[status_id] != 3:
                            ugv_stop_step_phase_list[-1].append(status_id)
                            ugv_stop_step_phase_list.append([])
                        if status_list[status_id - 1] != 2 and status_list[status_id] == 2:
                            ugv_stop_step_phase_list[-1].append(status_id)

                final_ugv_stop_step_phase_list = []
                for ugv_stop_step_phase in ugv_stop_step_phase_list:
                    tmp_phase = []
                    if len(ugv_stop_step_phase) == 1:
                        for step_id in range(ugv_stop_step_phase[0], len(status_list) + 1):
                            tmp_phase.append(step_id)
                        final_ugv_stop_step_phase_list.append(copy.deepcopy(tmp_phase))
                    elif len(ugv_stop_step_phase) == 2:
                        for step_id in range(ugv_stop_step_phase[0], ugv_stop_step_phase[1]):
                            tmp_phase.append(step_id)
                        final_ugv_stop_step_phase_list.append(copy.deepcopy(tmp_phase))
                for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                    step_id = final_ugv_stop_step_phase[0]
                    circ = plt.Circle((ugv_coordx_list[step_id], ugv_coordy_list[step_id]),
                                      self.env_conf['uav_ugv_max_dis'] + self.env_conf['uav_sensing_range'],
                                      color=self.color_list[UGV_UAVs_Group_id],
                                      alpha=0.2, fill=False)
                    ax.add_patch(circ)

                for ugv_pos_id in range(len(ugv_coordx_list)):
                    stop_flag = False
                    for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                        if ugv_pos_id in final_ugv_stop_step_phase:
                            stop_flag = True
                            break
                    if stop_flag:
                        for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                            if ugv_pos_id == final_ugv_stop_step_phase[0]:
                                plt.annotate(
                                    text=str(ugv_id) + '-' + str(ugv_pos_id) + '~' + str(final_ugv_stop_step_phase[-1]),
                                    xy=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]),
                                    xytext=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]), fontsize=0.5)
                                break
                    else:
                        plt.annotate(text=str(ugv_id) + '-' + str(ugv_pos_id),
                                     xy=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]),
                                     xytext=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]), fontsize=0.5)
            elif self.method_conf['ugv_trace_type'] == 'stops_net':
                cur_stop_id_list = UGV_UAVs_Group.ugv.episode_log_info_dict['cur_stop_id_list']
                for item_id, cur_stop_id in enumerate(cur_stop_id_list[:-1]):
                    start_stop_id = cur_stop_id
                    goal_stop_id = cur_stop_id_list[item_id + 1]
                    stops_net_SP_key = str(start_stop_id) + '_' + str(goal_stop_id)
                    shortest_path = env.stops_net_SP_dict[stops_net_SP_key]['shortest_path']
                    for sub_item_id, stop_id in enumerate(shortest_path[:-1]):
                        ugv_coordx_list = []
                        ugv_coordy_list = []
                        ugv_coordx_list.append(env.stops_net_dict[stop_id]['coordxy'][0])
                        ugv_coordy_list.append(env.stops_net_dict[stop_id]['coordxy'][1])
                        if shortest_path[sub_item_id + 1] != stop_id:
                            for mid_road_node_id in env.stops_net_dict[stop_id]['next_node2mid_road_id_list_dict'][
                                shortest_path[sub_item_id + 1]]:
                                road_node_coordx, road_node_coordy = env.roads_net_dict[mid_road_node_id]['coordxy']
                                ugv_coordx_list.append(road_node_coordx)
                                ugv_coordy_list.append(road_node_coordy)
                        ugv_coordx_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][0])
                        ugv_coordy_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][1])
                        plt.plot(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id],
                                 linewidth=2, alpha=0.5)
                ugv_coordx_list = []
                ugv_coordy_list = []
                for stop_id in cur_stop_id_list:
                    ugv_coordx_list.append(env.stops_net_dict[stop_id]['coordxy'][0])
                    ugv_coordy_list.append(env.stops_net_dict[stop_id]['coordxy'][1])
                plt.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id],
                            marker=self.marker_list[0], s=10)

                status_list = UGV_UAVs_Group.episode_log_info_dict['status']
                ugv_stop_step_phase_list = []
                ugv_stop_step_phase_list.append([])
                for status_id, status in enumerate(status_list):
                    if status_id == 0:
                        if status == 2:
                            ugv_stop_step_phase_list[-1].append(status_id)
                    else:
                        if status_list[status_id - 1] == 3 and status_list[status_id] != 3:
                            ugv_stop_step_phase_list[-1].append(status_id)
                            ugv_stop_step_phase_list.append([])
                        if status_list[status_id - 1] != 2 and status_list[status_id] == 2:
                            ugv_stop_step_phase_list[-1].append(status_id)

                final_ugv_stop_step_phase_list = []
                for ugv_stop_step_phase in ugv_stop_step_phase_list:
                    tmp_phase = []
                    if len(ugv_stop_step_phase) == 1:
                        for step_id in range(ugv_stop_step_phase[0], len(status_list) + 1):
                            tmp_phase.append(step_id)
                        final_ugv_stop_step_phase_list.append(copy.deepcopy(tmp_phase))
                    elif len(ugv_stop_step_phase) == 2:
                        for step_id in range(ugv_stop_step_phase[0], ugv_stop_step_phase[1]):
                            tmp_phase.append(step_id)
                        final_ugv_stop_step_phase_list.append(copy.deepcopy(tmp_phase))
                for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                    step_id = final_ugv_stop_step_phase[0]
                    circ = plt.Circle((ugv_coordx_list[step_id], ugv_coordy_list[step_id]),
                                      self.env_conf['uav_ugv_max_dis'] + self.env_conf['uav_sensing_range'],
                                      color=self.color_list[UGV_UAVs_Group_id],
                                      alpha=0.2, fill=False)
                    ax.add_patch(circ)

                for ugv_pos_id in range(len(ugv_coordx_list)):
                    stop_flag = False
                    for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                        if ugv_pos_id in final_ugv_stop_step_phase:
                            stop_flag = True
                            break
                    if stop_flag:
                        for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                            if ugv_pos_id == final_ugv_stop_step_phase[0]:
                                plt.annotate(
                                    text=str(ugv_id) + '-' + str(ugv_pos_id) + '~' + str(final_ugv_stop_step_phase[-1]),
                                    xy=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]),
                                    xytext=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]), fontsize=0.5)
                                break
                    else:
                        plt.annotate(text=str(ugv_id) + '-' + str(ugv_pos_id),
                                     xy=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]),
                                     xytext=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]), fontsize=0.5)

            # draw uav trace
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav = UGV_UAVs_Group.uav_list[uav_id]
                pos_list = uav.episode_log_info_dict['final_pos']
                uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][uav_sub_rollout_id].sub_episode_buffer_list:
                    uav_coordx_list = []
                    uav_coordy_list = []
                    for step_id in sub_episode_buffer['step_id_s']:
                        uav_coordx_list.append(pos_list[step_id][0])
                        uav_coordy_list.append(pos_list[step_id][1])
                    # plt.plot(uav_coordx_list, uav_coordy_list,
                    #          color=self.color_list[self.env_conf['uav_num_each_group'] + uav_id], linewidth=1,
                    #          alpha=0.5)
                    # plt.scatter(uav_coordx_list, uav_coordy_list,
                    #             color=self.color_list[self.env_conf['uav_num_each_group'] + uav_id],
                    #             marker=self.marker_list[1 + uav_id], s=0.5)
                    plt.plot(uav_coordx_list, uav_coordy_list,
                             color=self.color_list[UGV_UAVs_Group_id], linewidth=1,
                             alpha=0.5)
                    plt.scatter(uav_coordx_list, uav_coordy_list,
                                color=self.color_list[UGV_UAVs_Group_id],
                                marker=self.marker_list[1 + uav_id], s=0.5)

        # draw buildings
        obstacles_file_path = os.path.join(self.dataset_conf['dataset_path'], 'obstacles.shp')
        obstacles_file = shapefile.Reader(obstacles_file_path)

        border_shape = obstacles_file
        border = border_shape.shapes()

        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.dataset_conf['zone_id'],
                                       self.dataset_conf['ball_id'])
            coordxs = (lons - self.dataset_conf['lon_min']) * self.dataset_conf['coordx_per_lon']
            coordys = (lats - self.dataset_conf['lat_min']) * self.dataset_conf['coordy_per_lat']
            plt.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

        # draw border
        border_file_path = os.path.join(self.dataset_conf['dataset_path'], 'border.shp')
        border_file = shapefile.Reader(border_file_path)
        border_shape = border_file
        border = border_shape.shapes()
        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.dataset_conf['zone_id'],
                                       self.dataset_conf['ball_id'])
            coordxs = (lons - self.dataset_conf['lon_min']) * self.dataset_conf['coordx_per_lon']
            coordys = (lats - self.dataset_conf['lat_min']) * self.dataset_conf['coordy_per_lat']
            plt.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

        # draw pois
        poi_coordx_array = env.poi_coordxy_array[:, 0]
        poi_coordy_array = env.poi_coordxy_array[:, 1]
        poi_final_value_norm = env.poi_cur_value_array / self.env_conf['poi_value_max']
        plt.scatter(poi_coordx_array, poi_coordy_array, c=poi_final_value_norm, vmin=0, vmax=1, cmap=cm, s=100,
                    zorder=5)

        # draw roads
        for road_node_id in env.roads_net_dict:
            road_node_coordx, road_node_coordy = env.roads_net_dict[road_node_id]['coordxy']
            next_node_list = env.roads_net_dict[road_node_id]['next_node_list']
            for next_node_id in next_node_list:
                next_road_node_coordx, next_road_node_coordy = env.roads_net_dict[next_node_id]['coordxy']
                plt.plot([road_node_coordx, next_road_node_coordx], [road_node_coordy, next_road_node_coordy],
                         color='green', label='fungis', linewidth=1, alpha=0.2)
            # plt.scatter(road_node_coordx, road_node_coordy, c='black', alpha=0.2, s=10, zorder=1)
            # plt.annotate(text=str(road_node_id), xy=(road_node_coordx, road_node_coordy),
            #              xytext=(road_node_coordx, road_node_coordy), fontsize=0.5)

        eff = np.round(self.episode_metrics_result['eff'][iter_id], 2)
        fairness = np.round(self.episode_metrics_result['fairness'][iter_id], 2)
        dcr = np.round(self.episode_metrics_result['dcr'][iter_id], 2)
        cor = np.round(self.episode_metrics_result['cor'][iter_id], 2)
        hit = np.round(self.episode_metrics_result['hit'][iter_id], 2)
        ec = np.round(self.episode_metrics_result['ec'][iter_id], 2)
        ecr = np.round(self.episode_metrics_result['ecr'][iter_id], 2)

        title_str = 'iter: ' + str(iter_id) \
                    + ' eff: ' + str(eff) \
                    + ' fairness: ' + str(fairness) \
                    + ' dcr: ' + str(dcr) \
                    + ' ecr: ' + str(ecr) \
                    + '\n' \
                    + ' cor: ' + str(cor) \
                    + ' hit: ' + str(hit) \
                    + ' ec: ' + str(ec)

        plt.suptitle(title_str)

        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
        # plt.rcParams['figure.figsize'] = (5, 5)
        # plt.subplots_adjust(left=10, bottom=10, right=11, top=11, hspace=0, wspace=0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        my_axis = plt.gca()
        my_axis.spines['top'].set_linewidth(10)
        my_axis.spines['bottom'].set_linewidth(10)
        my_axis.spines['left'].set_linewidth(10)
        my_axis.spines['right'].set_linewidth(10)
        # plt.tight_layout()
        if mode == 'train':
            fig_dir_path = os.path.join(self.log_path, 'trace')
            if not os.path.exists(fig_dir_path):
                os.makedirs(fig_dir_path)
            fig_path = os.path.join(fig_dir_path, 'trace_iter_id' + str(iter_id) + '.png')
            Fig.savefig(fig_path)
            # Fig.savefig(fig_path, dpi=1300)
        elif mode == 'test':
            fig_dir_path = os.path.join(self.root_log_path, 'trace')
            if not os.path.exists(fig_dir_path):
                os.makedirs(fig_dir_path)
            fig_path = os.path.join(fig_dir_path, 'subp_test_id' + str(subp_id) + '.png')
            Fig.savefig(fig_path)
            # Fig.savefig(fig_path, dpi=1300)
        plt.close()

    def record_trace2(self, iter_id, env, sub_rollout_manager, mode='train', subp_id=None):
        mpl.style.use('default')
        # Fig = plt.figure(figsize=(10, 10))
        Fig, ax = plt.subplots(figsize=(10, 10))
        plt.xlim(xmin=0, xmax=self.dataset_conf['coordx_max'])
        plt.ylim(ymin=0, ymax=self.dataset_conf['coordy_max'])
        # plt.grid(True, linestyle='-.', color='r')

        cm = plt.cm.get_cmap('RdYlBu_r')
        # draw buildings
        obstacles_file_path = os.path.join(self.dataset_conf['dataset_path'], 'obstacles.shp')
        obstacles_file = shapefile.Reader(obstacles_file_path)

        border_shape = obstacles_file
        border = border_shape.shapes()

        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.dataset_conf['zone_id'],
                                       self.dataset_conf['ball_id'])
            coordxs = (lons - self.dataset_conf['lon_min']) * self.dataset_conf['coordx_per_lon']
            coordys = (lats - self.dataset_conf['lat_min']) * self.dataset_conf['coordy_per_lat']
            plt.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

        # draw border
        border_file_path = os.path.join(self.dataset_conf['dataset_path'], 'border.shp')
        border_file = shapefile.Reader(border_file_path)
        border_shape = border_file
        border = border_shape.shapes()
        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.dataset_conf['zone_id'],
                                       self.dataset_conf['ball_id'])
            coordxs = (lons - self.dataset_conf['lon_min']) * self.dataset_conf['coordx_per_lon']
            coordys = (lats - self.dataset_conf['lat_min']) * self.dataset_conf['coordy_per_lat']
            plt.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色
        # draw roads
        for road_node_id in env.roads_net_dict:
            road_node_coordx, road_node_coordy = env.roads_net_dict[road_node_id]['coordxy']
            next_node_list = env.roads_net_dict[road_node_id]['next_node_list']
            for next_node_id in next_node_list:
                next_road_node_coordx, next_road_node_coordy = env.roads_net_dict[next_node_id]['coordxy']
                plt.plot([road_node_coordx, next_road_node_coordx], [road_node_coordy, next_road_node_coordy],
                         color='grey', label='fungis', linewidth=10, alpha=0.2)
            # plt.scatter(road_node_coordx, road_node_coordy, c='black', alpha=0.2, s=10, zorder=1)
            # plt.annotate(text=str(road_node_id), xy=(road_node_coordx, road_node_coordy),
            #              xytext=(road_node_coordx, road_node_coordy), fontsize=0.5)
        # draw pois
        poi_coordx_array = env.poi_coordxy_array[:, 0]
        poi_coordy_array = env.poi_coordxy_array[:, 1]
        poi_final_value_norm = env.poi_cur_value_array / self.env_conf['poi_value_max']
        plt.scatter(poi_coordx_array, poi_coordy_array, c=poi_final_value_norm, vmin=0, vmax=1, cmap=cm, s=100,
                    zorder=5)
        # draw trace
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            UGV_UAVs_Group = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id]
            # draw UGV trace
            ugv_id = UGV_UAVs_Group_id
            if self.method_conf['ugv_trace_type'] == 'roads_net':
                final_passed_road_node_id_list = \
                    UGV_UAVs_Group.ugv.episode_log_info_dict['final_passed_road_node_id_list'][
                        -1]
                road_pos_list = UGV_UAVs_Group.ugv.episode_log_info_dict['final_road_pos']
                for item_id, passed_road_node_id_list in enumerate(final_passed_road_node_id_list):
                    ugv_coordx_list = []
                    ugv_coordy_list = []
                    ugv_coordx_list.append(env.road_pos2pos(road_pos_list[item_id])[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos_list[item_id])[1])
                    for passed_road_node_id in passed_road_node_id_list:
                        ugv_coordx_list.append(env.roads_net_dict[passed_road_node_id]['coordxy'][0])
                        ugv_coordy_list.append(env.roads_net_dict[passed_road_node_id]['coordxy'][1])
                    ugv_coordx_list.append(env.road_pos2pos(road_pos_list[item_id + 1])[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos_list[item_id + 1])[1])
                    plt.plot(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)], linewidth=20,
                             alpha=0.5)

                ugv_coordx_list = []
                ugv_coordy_list = []
                for road_pos in road_pos_list:
                    ugv_coordx_list.append(env.road_pos2pos(road_pos)[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos)[1])
                plt.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                            marker=self.marker_list[0], s=10)

                # status_list = UGV_UAVs_Group.episode_log_info_dict['status']
                # ugv_stop_step_phase_list = []
                # ugv_stop_step_phase_list.append([])
                # for status_id, status in enumerate(status_list):
                #     if status_id == 0:
                #         if status == 2:
                #             ugv_stop_step_phase_list[-1].append(status_id)
                #     else:
                #         if status_list[status_id - 1] == 3 and status_list[status_id] != 3:
                #             ugv_stop_step_phase_list[-1].append(status_id)
                #             ugv_stop_step_phase_list.append([])
                #         if status_list[status_id - 1] != 2 and status_list[status_id] == 2:
                #             ugv_stop_step_phase_list[-1].append(status_id)
                #
                # final_ugv_stop_step_phase_list = []
                # for ugv_stop_step_phase in ugv_stop_step_phase_list:
                #     tmp_phase = []
                #     if len(ugv_stop_step_phase) == 1:
                #         for step_id in range(ugv_stop_step_phase[0], len(status_list) + 1):
                #             tmp_phase.append(step_id)
                #         final_ugv_stop_step_phase_list.append(copy.deepcopy(tmp_phase))
                #     elif len(ugv_stop_step_phase) == 2:
                #         for step_id in range(ugv_stop_step_phase[0], ugv_stop_step_phase[1]):
                #             tmp_phase.append(step_id)
                #         final_ugv_stop_step_phase_list.append(copy.deepcopy(tmp_phase))
                # for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                #     step_id = final_ugv_stop_step_phase[0]
                #     circ = plt.Circle((ugv_coordx_list[step_id], ugv_coordy_list[step_id]),
                #                       self.env_conf['uav_ugv_max_dis'] + self.env_conf['uav_sensing_range'],
                #                       color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                #                       alpha=0.2, fill=False)
                #     ax.add_patch(circ)

                # for ugv_pos_id in range(len(ugv_coordx_list)):
                #     stop_flag = False
                #     for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                #         if ugv_pos_id in final_ugv_stop_step_phase:
                #             stop_flag = True
                #             break
                #     if stop_flag:
                #         for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                #             if ugv_pos_id == final_ugv_stop_step_phase[0]:
                #                 plt.annotate(
                #                     text=str(ugv_id) + '-' + str(ugv_pos_id) + '~' + str(final_ugv_stop_step_phase[-1]),
                #                     xy=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]),
                #                     xytext=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]), fontsize=0.5)
                #                 break
                #     else:
                #         plt.annotate(text=str(ugv_id) + '-' + str(ugv_pos_id),
                #                      xy=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]),
                #                      xytext=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]), fontsize=0.5)
            elif self.method_conf['ugv_trace_type'] == 'stops_net':
                cur_stop_id_list = UGV_UAVs_Group.ugv.episode_log_info_dict['cur_stop_id_list']
                for item_id, cur_stop_id in enumerate(cur_stop_id_list[:-1]):
                    start_stop_id = cur_stop_id
                    goal_stop_id = cur_stop_id_list[item_id + 1]
                    stops_net_SP_key = str(start_stop_id) + '_' + str(goal_stop_id)
                    shortest_path = env.stops_net_SP_dict[stops_net_SP_key]['shortest_path']
                    for sub_item_id, stop_id in enumerate(shortest_path[:-1]):
                        ugv_coordx_list = []
                        ugv_coordy_list = []
                        ugv_coordx_list.append(env.stops_net_dict[stop_id]['coordxy'][0])
                        ugv_coordy_list.append(env.stops_net_dict[stop_id]['coordxy'][1])
                        if shortest_path[sub_item_id + 1] != stop_id:
                            for mid_road_node_id in env.stops_net_dict[stop_id]['next_node2mid_road_id_list_dict'][
                                shortest_path[sub_item_id + 1]]:
                                road_node_coordx, road_node_coordy = env.roads_net_dict[mid_road_node_id]['coordxy']
                                ugv_coordx_list.append(road_node_coordx)
                                ugv_coordy_list.append(road_node_coordy)
                        ugv_coordx_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][0])
                        ugv_coordy_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][1])
                        plt.plot(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                                 linewidth=20, alpha=0.5)
                ugv_coordx_list = []
                ugv_coordy_list = []
                for stop_id in cur_stop_id_list:
                    ugv_coordx_list.append(env.stops_net_dict[stop_id]['coordxy'][0])
                    ugv_coordy_list.append(env.stops_net_dict[stop_id]['coordxy'][1])
                plt.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                            marker=self.marker_list[0], s=10)

                # status_list = UGV_UAVs_Group.episode_log_info_dict['status']
                # ugv_stop_step_phase_list = []
                # ugv_stop_step_phase_list.append([])
                # for status_id, status in enumerate(status_list):
                #     if status_id == 0:
                #         if status == 2:
                #             ugv_stop_step_phase_list[-1].append(status_id)
                #     else:
                #         if status_list[status_id - 1] == 3 and status_list[status_id] != 3:
                #             ugv_stop_step_phase_list[-1].append(status_id)
                #             ugv_stop_step_phase_list.append([])
                #         if status_list[status_id - 1] != 2 and status_list[status_id] == 2:
                #             ugv_stop_step_phase_list[-1].append(status_id)
                #
                # final_ugv_stop_step_phase_list = []
                # for ugv_stop_step_phase in ugv_stop_step_phase_list:
                #     tmp_phase = []
                #     if len(ugv_stop_step_phase) == 1:
                #         for step_id in range(ugv_stop_step_phase[0], len(status_list) + 1):
                #             tmp_phase.append(step_id)
                #         final_ugv_stop_step_phase_list.append(copy.deepcopy(tmp_phase))
                #     elif len(ugv_stop_step_phase) == 2:
                #         for step_id in range(ugv_stop_step_phase[0], ugv_stop_step_phase[1]):
                #             tmp_phase.append(step_id)
                #         final_ugv_stop_step_phase_list.append(copy.deepcopy(tmp_phase))
                # for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                #     step_id = final_ugv_stop_step_phase[0]
                #     circ = plt.Circle((ugv_coordx_list[step_id], ugv_coordy_list[step_id]),
                #                       self.env_conf['uav_ugv_max_dis'] + self.env_conf['uav_sensing_range'],
                #                       color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                #                       alpha=0.2, fill=False)
                #     ax.add_patch(circ)

                # for ugv_pos_id in range(len(ugv_coordx_list)):
                #     stop_flag = False
                #     for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                #         if ugv_pos_id in final_ugv_stop_step_phase:
                #             stop_flag = True
                #             break
                #     if stop_flag:
                #         for final_ugv_stop_step_phase in final_ugv_stop_step_phase_list:
                #             if ugv_pos_id == final_ugv_stop_step_phase[0]:
                #                 plt.annotate(
                #                     text=str(ugv_id) + '-' + str(ugv_pos_id) + '~' + str(final_ugv_stop_step_phase[-1]),
                #                     xy=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]),
                #                     xytext=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]), fontsize=0.5)
                #                 break
                #     else:
                #         plt.annotate(text=str(ugv_id) + '-' + str(ugv_pos_id),
                #                      xy=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]),
                #                      xytext=(ugv_coordx_list[ugv_pos_id], ugv_coordy_list[ugv_pos_id]), fontsize=0.5)

            # draw uav trace
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav = UGV_UAVs_Group.uav_list[uav_id]
                pos_list = uav.episode_log_info_dict['final_pos']
                uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                    uav_sub_rollout_id].sub_episode_buffer_list:
                    uav_coordx_list = []
                    uav_coordy_list = []
                    for step_id in sub_episode_buffer['step_id_s']:
                        uav_coordx_list.append(pos_list[step_id][0])
                        uav_coordy_list.append(pos_list[step_id][1])
                    plt.plot(uav_coordx_list, uav_coordy_list,
                             color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)], linewidth=5,
                             alpha=0.5)
                    plt.scatter(uav_coordx_list, uav_coordy_list,
                                color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                                marker=self.marker_list[(1 + uav_id) % len(self.marker_list)], s=0.5)

        eff = np.round(self.episode_metrics_result['eff'][iter_id], 2)
        fairness = np.round(self.episode_metrics_result['fairness'][iter_id], 2)
        cor = np.round(self.episode_metrics_result['cor'][iter_id], 2)
        dcr = np.round(self.episode_metrics_result['dcr'][iter_id], 2)
        hit = np.round(self.episode_metrics_result['hit'][iter_id], 2)
        ec = np.round(self.episode_metrics_result['ec'][iter_id], 2)
        ecr = np.round(self.episode_metrics_result['ecr'][iter_id], 2)

        title_str = 'iter: ' + str(iter_id) \
                    + ' eff: ' + str(eff) \
                    + ' fairness: ' + str(fairness) \
                    + ' cor: ' + str(cor) \
                    + ' dcr: ' + str(dcr) \
                    + ' ecr: ' + str(ecr) \
                    + '\n' \
                    + ' hit: ' + str(hit) \
                    + ' ec: ' + str(ec)

        # plt.suptitle(title_str)

        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
        # plt.rcParams['figure.figsize'] = (5, 5)
        # plt.subplots_adjust(left=10, bottom=10, right=11, top=11, hspace=0, wspace=0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        my_axis = plt.gca()
        my_axis.spines['top'].set_linewidth(10)
        my_axis.spines['bottom'].set_linewidth(10)
        my_axis.spines['left'].set_linewidth(10)
        my_axis.spines['right'].set_linewidth(10)
        # plt.tight_layout()
        if mode == 'train':
            fig_dir_path = os.path.join(self.log_path, 'trace')
            if not os.path.exists(fig_dir_path):
                os.makedirs(fig_dir_path)
            fig_path = os.path.join(fig_dir_path, 'trace_iter_id' + str(iter_id) + '.png')
            Fig.savefig(fig_path)
            # Fig.savefig(fig_path, dpi=1300)
        elif mode == 'test':
            fig_dir_path = os.path.join(self.root_log_path, 'trace')
            if not os.path.exists(fig_dir_path):
                os.makedirs(fig_dir_path)
            fig_path = os.path.join(fig_dir_path, 'dcr_' + str(dcr) + '_subp_test_id' + str(subp_id) + '.png')
            Fig.savefig(fig_path)
            # Fig.savefig(fig_path, dpi=1300)
        plt.close()


