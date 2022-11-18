import copy
import os

import numpy as np

from util import *


class SubLog:
    def __init__(self, process_id=None, log_path=None):
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.episode_metrics_result = {}
        self.episode_metrics_result['eff'] = []
        self.episode_metrics_result['eff2'] = []
        self.episode_metrics_result['fairness'] = []
        self.episode_metrics_result['fairness2'] = []
        self.episode_metrics_result['dcr'] = []
        self.episode_metrics_result['hit'] = []
        self.episode_metrics_result['ec'] = []
        self.episode_metrics_result['ecr'] = []
        self.episode_metrics_result['cor'] = []
        self.episode_metrics_result['cor2'] = []

        self.root_log_path = log_path
        self.log_path = os.path.join(log_path, 'process_' + str(process_id))

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.episode_log_info_dict_dir_path = os.path.join(self.log_path, 'episode_log_info_dict')
        if not os.path.exists(self.episode_log_info_dict_dir_path):
            os.makedirs(self.episode_log_info_dict_dir_path)

        # for draw
        self.color_list = ['C' + str(color_id) for color_id in range(10)]
        self.marker_list = ['o', '.', 'v', '^', '<', '>', '1', '2', '3', '4']

    def record_sub_rollout_dict(self, sub_rollout_manager):
        sub_rollout_dict_path = os.path.join(self.log_path, 'sub_rollout_dict.npy')
        np.save(sub_rollout_dict_path, sub_rollout_manager.sub_rollout_dict)

    def record_episode_log_info_dict(self, iter_id, env):
        episode_log_info_dict = {}
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            episode_log_info_dict[UGV_UAVs_Group_id] = {}
            episode_log_info_dict[UGV_UAVs_Group_id]['UGV_UAVs_Group'] = copy.deepcopy(
                env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].episode_log_info_dict)
            episode_log_info_dict[UGV_UAVs_Group_id]['UGV'] = copy.deepcopy(
                env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].ugv.episode_log_info_dict)
            episode_log_info_dict[UGV_UAVs_Group_id]['UAVs'] = [copy.deepcopy(uav.episode_log_info_dict) for uav in
                                                                env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list]
        episode_log_info_dict_path = os.path.join(self.episode_log_info_dict_dir_path,
                                                  'episode_log_info_dict_' + str(iter_id) + '.npy')
        np.save(episode_log_info_dict_path, episode_log_info_dict)

    def record_errors(self, report_str):
        self._report_path = self.log_path + '/errors.txt'
        f = open(self._report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()

    def gen_metrics_result(self, iter_id, env):
        # fairness
        fairness = 0.0
        final_poi_visit_time = env.episode_log_info_dict['final_poi_visit_time'][-1]
        square_of_sum = np.square(np.sum(final_poi_visit_time))
        sum_of_square = np.sum(np.square(final_poi_visit_time))
        if sum_of_square > 1e-5:
            fairness = square_of_sum / sum_of_square / final_poi_visit_time.shape[0]
        if len(self.episode_metrics_result['fairness']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['fairness'].append(fairness)

        # fairness2
        fairness2 = 0.0
        # collect_poi_ratio = env.poi_cur_value_array / env.poi_init_value_array
        # square_of_sum = np.square(np.sum(collect_poi_ratio))
        # sum_of_square = np.sum(np.square(collect_poi_ratio))
        final_poi_visit_time = np.clip(env.episode_log_info_dict['final_poi_visit_time'][-1], 0, 2)
        square_of_sum = np.square(np.sum(final_poi_visit_time))
        sum_of_square = np.sum(np.square(final_poi_visit_time))
        if sum_of_square > 1e-5:
            fairness2 = square_of_sum / sum_of_square / final_poi_visit_time.shape[0]
        if len(self.episode_metrics_result['fairness2']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['fairness2'].append(fairness2)

        # data_collection_ratio (dcr)
        dcr = np.sum(env.poi_init_value_array - env.poi_cur_value_array) / np.sum(env.poi_init_value_array)
        if len(self.episode_metrics_result['dcr']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['dcr'].append(dcr)

        # hit
        hit = env.final_total_hit
        if len(self.episode_metrics_result['hit']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['hit'].append(hit)

        # energy_consumption (ec)
        ec = env.final_energy_consumption
        if len(self.episode_metrics_result['ec']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['ec'].append(ec)

        # energy_consumption_ratio (ecr)
        ecr = ec / env.ec_upper_bound
        if len(self.episode_metrics_result['ecr']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['ecr'].append(ecr)

        # UGV_UAV_cooperation_ratio (cor)
        if env.final_total_fly_time > 0:
            cor = env.final_total_collect_data_time / env.final_total_fly_time
        else:
            cor = 0
        if len(self.episode_metrics_result['cor']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['cor'].append(cor)

        # UGV_UAV_cooperation_ratio2 (cor2)
        if env.final_total_relax_time > 0:
            cor2 = env.final_total_eff_relax_time / env.final_total_relax_time
        else:
            cor2 = 0
        if len(self.episode_metrics_result['cor2']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['cor2'].append(cor2)

        # eff
        eff = 0.0
        if ecr > min_value:
            eff = fairness * dcr * cor / ecr
        if len(self.episode_metrics_result['eff']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['eff'].append(eff)

        # eff2
        eff2 = 0.0
        if ecr > min_value:
            eff2 = fairness2 * dcr * cor2 / ecr
        if len(self.episode_metrics_result['eff2']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['eff2'].append(eff2)

    def record_metrics_result(self):
        np.save(self.log_path + '/episode_metrics_result.npy', self.episode_metrics_result)

    def record_trace_se(self, iter_id, env, sub_rollout_manager):
        mpl.style.use('default')
        Fig = plt.figure(figsize=(10, 5))
        ax1 = Fig.add_subplot(121)
        ax2 = Fig.add_subplot(122)

        ax1.set_xlim(xmin=0, xmax=self.dataset_conf['coordx_max'])
        ax1.set_ylim(ymin=0, ymax=self.dataset_conf['coordy_max'])
        # ax1.grid(True, linestyle='-.', color='r')
        ax2.set_xlim(xmin=0, xmax=self.dataset_conf['coordx_max'])
        ax2.set_ylim(ymin=0, ymax=self.dataset_conf['coordy_max'])
        # ax2.grid(True, linestyle='-.', color='r')

        cm = plt.cm.get_cmap('RdYlBu_r')

        # draw trace
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            UGV_UAVs_Group = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id]
            # draw UGV trace
            ugv_id = UGV_UAVs_Group_id
            final_passed_road_node_id_list = UGV_UAVs_Group.ugv.episode_log_info_dict['final_passed_road_node_id_list'][
                -1]
            start_road_pos = UGV_UAVs_Group.ugv.episode_log_info_dict['final_road_pos'][0]
            end_road_pos = UGV_UAVs_Group.ugv.episode_log_info_dict['final_road_pos'][-1]
            ugv_coordx_list = []
            ugv_coordy_list = []
            ugv_coordx_list.append(env.road_pos2pos(start_road_pos)[0])
            ugv_coordy_list.append(env.road_pos2pos(start_road_pos)[1])
            for road_node_id in final_passed_road_node_id_list:
                ugv_coordx_list.append(env.roads_net_dict[road_node_id]['coordxy'][0])
                ugv_coordy_list.append(env.roads_net_dict[road_node_id]['coordxy'][1])
            ugv_coordx_list.append(env.road_pos2pos(end_road_pos)[0])
            ugv_coordy_list.append(env.road_pos2pos(end_road_pos)[1])
            ax2.plot(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                     linewidth=2, alpha=0.5)

            road_pos_list = UGV_UAVs_Group.ugv.episode_log_info_dict['final_road_pos']
            ugv_coordx_list = []
            ugv_coordy_list = []
            for road_pos in road_pos_list:
                ugv_coordx_list.append(env.road_pos2pos(road_pos)[0])
                ugv_coordy_list.append(env.road_pos2pos(road_pos)[1])
            ax2.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                        marker=self.marker_list[0], s=1)

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
                    ax2.plot(uav_coordx_list, uav_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)], linewidth=1,
                             alpha=0.5)
                    ax2.scatter(uav_coordx_list, uav_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                                marker=self.marker_list[(1 + uav_id) % len(self.marker_list)], s=0.5)

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
            ax1.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色
            ax2.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

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
            ax1.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色
            ax2.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

        # draw pois
        poi_coordx_array = env.poi_coordxy_array[:, 0]
        poi_coordy_array = env.poi_coordxy_array[:, 1]
        poi_init_value_norm = env.poi_init_value_array / self.env_conf['poi_value_max']
        poi_final_value_norm = env.poi_cur_value_array / self.env_conf['poi_value_max']
        ax1.scatter(poi_coordx_array, poi_coordy_array, c=poi_init_value_norm, vmin=0, vmax=1, cmap=cm, s=100, zorder=5)
        ax2.scatter(poi_coordx_array, poi_coordy_array, c=poi_final_value_norm, vmin=0, vmax=1, cmap=cm, s=100,
                    zorder=5)

        # draw roads
        for road_node_id in env.roads_net_dict:
            road_node_coordx, road_node_coordy = env.roads_net_dict[road_node_id]['coordxy']
            next_node_list = env.roads_net_dict[road_node_id]['next_node_list']
            for next_node_id in next_node_list:
                next_road_node_coordx, next_road_node_coordy = env.roads_net_dict[next_node_id]['coordxy']
                ax1.plot([road_node_coordx, next_road_node_coordx], [road_node_coordy, next_road_node_coordy],
                         color='green', label='fungis', linewidth=1)
                ax2.plot([road_node_coordx, next_road_node_coordx], [road_node_coordy, next_road_node_coordy],
                         color='green', label='fungis', linewidth=1)
            # plt.scatter(road_node_coordx, road_node_coordy, c='black', alpha=0.2, s=10, zorder=1)
            # plt.annotate(text=str(road_node_id), xy=(road_node_coordx, road_node_coordy),
            #              xytext=(road_node_coordx, road_node_coordy), fontsize=0.5)

        eff = np.round(self.episode_metrics_result['eff'][iter_id], 2)
        fairness = np.round(self.episode_metrics_result['fairness'][iter_id], 2)
        dcr = np.round(self.episode_metrics_result['dcr'][iter_id], 2)
        hit = np.round(self.episode_metrics_result['hit'][iter_id], 2)
        ec = np.round(self.episode_metrics_result['ec'][iter_id], 2)
        ecr = np.round(self.episode_metrics_result['ecr'][iter_id], 2)

        title_str = 'iter: ' + str(iter_id) \
                    + ' eff: ' + str(eff) \
                    + ' fairness: ' + str(fairness) \
                    + ' dcr: ' + str(dcr) \
                    + ' ecr: ' + str(ecr) \
                    + '\n' \
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

        fig_dir_path = os.path.join(self.log_path, 'trace_se')
        if not os.path.exists(fig_dir_path):
            os.makedirs(fig_dir_path)
        fig_path = os.path.join(fig_dir_path, 'trace_se_iter_id' + str(iter_id) + '.png')
        Fig.savefig(fig_path)
        # Fig.savefig(fig_path, dpi=1300)
        plt.close()

    def record_trace_old(self, iter_id, env, sub_rollout_manager, mode='train', subp_id=None):
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
                    plt.plot(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)], linewidth=2,
                             alpha=0.5)

                ugv_coordx_list = []
                ugv_coordy_list = []
                for road_pos in road_pos_list:
                    ugv_coordx_list.append(env.road_pos2pos(road_pos)[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos)[1])
                plt.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
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
                                      color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
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
                        plt.plot(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                                 linewidth=2, alpha=0.5)
                ugv_coordx_list = []
                ugv_coordy_list = []
                for stop_id in cur_stop_id_list:
                    ugv_coordx_list.append(env.stops_net_dict[stop_id]['coordxy'][0])
                    ugv_coordy_list.append(env.stops_net_dict[stop_id]['coordxy'][1])
                plt.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
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
                                      color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
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
                for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                    uav_sub_rollout_id].sub_episode_buffer_list:
                    uav_coordx_list = []
                    uav_coordy_list = []
                    for step_id in sub_episode_buffer['step_id_s']:
                        uav_coordx_list.append(pos_list[step_id][0])
                        uav_coordy_list.append(pos_list[step_id][1])
                    plt.plot(uav_coordx_list, uav_coordy_list,
                             color=self.color_list[(self.env_conf['uav_num_each_group'] + uav_id) % len(self.color_list)], linewidth=1,
                             alpha=0.5)
                    plt.scatter(uav_coordx_list, uav_coordy_list,
                                color=self.color_list[(self.env_conf['uav_num_each_group'] + uav_id) % len(self.color_list)],
                                marker=self.marker_list[(1 + uav_id) % len(self.marker_list)], s=0.5)

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

        eff2 = np.round(self.episode_metrics_result['eff2'][iter_id], 2)
        fairness2 = np.round(self.episode_metrics_result['fairness2'][iter_id], 2)
        cor2 = np.round(self.episode_metrics_result['cor2'][iter_id], 2)
        dcr = np.round(self.episode_metrics_result['dcr'][iter_id], 2)
        hit = np.round(self.episode_metrics_result['hit'][iter_id], 2)
        ec = np.round(self.episode_metrics_result['ec'][iter_id], 2)
        ecr = np.round(self.episode_metrics_result['ecr'][iter_id], 2)

        title_str = 'iter: ' + str(iter_id) \
                    + ' eff2: ' + str(eff2) \
                    + ' fairness2: ' + str(fairness2) \
                    + ' cor2: ' + str(cor2) \
                    + ' dcr: ' + str(dcr) \
                    + ' ecr: ' + str(ecr) \
                    + '\n' \
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
            fig_path = os.path.join(fig_dir_path, 'dcr_' + str(dcr) + '_subp_test_id' + str(subp_id) + '.png')
            Fig.savefig(fig_path)
            # Fig.savefig(fig_path, dpi=1300)
        plt.close()

    def record_trace(self, iter_id, env, sub_rollout_manager, mode='train', subp_id=None):
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

    def record_trace_gif(self, iter_id, env, sub_rollout_manager, mode='train', subp_id=None):
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
                         color='grey', label='fungis', linewidth=1, alpha=0.2)
            # plt.scatter(road_node_coordx, road_node_coordy, c='black', alpha=0.2, s=10, zorder=1)
            # plt.annotate(text=str(road_node_id), xy=(road_node_coordx, road_node_coordy),
            #              xytext=(road_node_coordx, road_node_coordy), fontsize=0.5)
        # draw pois
        poi_coordx_array = env.poi_coordxy_array[:, 0]
        poi_coordy_array = env.poi_coordxy_array[:, 1]
        poi_final_value_norm = env.poi_cur_value_array / self.env_conf['poi_value_max']
        poi_scat = plt.scatter(poi_coordx_array, poi_coordy_array, c=poi_final_value_norm, vmin=0, vmax=1, cmap=cm,
                               s=100,
                               zorder=5)
        # define ugv scat and uav scat
        ugv_scat_list = []
        # uav_scat_dict = {}
        ugv_annotationbox_list = []
        uav_annotationbox_dict = {}
        uav_sensing_circle_dict = {}
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            UGV_UAVs_Group = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id]
            ugv_id = UGV_UAVs_Group_id
            road_pos_list = UGV_UAVs_Group.ugv.episode_log_info_dict['final_road_pos']
            ugv_coordx_list = []
            ugv_coordy_list = []
            ugv_coordx_list.append(env.road_pos2pos(road_pos_list[0])[0])
            ugv_coordy_list.append(env.road_pos2pos(road_pos_list[0])[1])
            # ugv_scat_list.append(plt.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
            #             marker=self.marker_list[0], s=100))
            ugv_scat_list.append(plt.scatter(np.array(ugv_coordx_list), np.array(ugv_coordy_list) - 50,
                                             color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                                             marker=r'$\mathtt{UGV%s}$' % (str(ugv_id)), s=1000, zorder=100))
            img_path = os.path.join(os.path.dirname(self.dataset_conf['dataset_path']), 'UGV_' + str(ugv_id) + '.png')
            ab = AnnotationBbox(OffsetImage(plt.imread(img_path), zoom=0.1), (ugv_coordx_list[0], ugv_coordy_list[0]),
                                xycoords='data', frameon=False)
            ab.set_animated(True)
            ax.add_artist(ab)
            ugv_annotationbox_list.append(ab)
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav = UGV_UAVs_Group.uav_list[uav_id]
                pos_list = uav.episode_log_info_dict['final_pos']
                uav_scat_id = str(ugv_id) + '-' + str(uav_id)
                # uav_scat_dict[uav_scat_id] = plt.scatter([pos_list[0][0]], [pos_list[0][1] - 25],
                #                                          color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                #                                          marker=r'$\mathtt{UAV}$',
                #                                          s=500, zorder=50)
                img_path = os.path.join(os.path.dirname(self.dataset_conf['dataset_path']),
                                        'UAV_' + str(ugv_id) + '.png')
                ab = AnnotationBbox(OffsetImage(plt.imread(img_path), zoom=0.05),
                                    (pos_list[0][0], pos_list[0][1]),
                                    xycoords='data', frameon=False)
                ab.set_animated(True)
                ab.set_zorder(100)
                ax.add_artist(ab)
                uav_annotationbox_dict[uav_scat_id] = ab
                circ = plt.Circle((pos_list[0][0], pos_list[0][1]),
                                  self.env_conf['uav_sensing_range'],
                                  color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                                  alpha=0.5, fill=False, zorder=100)
                circ.set_animated(True)
                ax.add_patch(circ)
                uav_sensing_circle_dict[uav_scat_id] = circ

        eff = np.round(self.episode_metrics_result['eff'][iter_id], 2)
        fairness = np.round(self.episode_metrics_result['fairness'][iter_id], 2)
        dcr = np.round(self.episode_metrics_result['dcr'][iter_id], 2)
        hit = np.round(self.episode_metrics_result['hit'][iter_id], 2)
        ec = np.round(self.episode_metrics_result['ec'][iter_id], 2)
        ecr = np.round(self.episode_metrics_result['ecr'][iter_id], 2)

        title_str = 'iter: ' + str(iter_id) \
                    + ' eff: ' + str(eff) \
                    + ' fairness: ' + str(fairness) \
                    + ' dcr: ' + str(dcr) \
                    + ' ecr: ' + str(ecr) \
                    + '\n' \
                    + ' hit: ' + str(hit) \
                    + ' ec: ' + str(ec)

        # plt.suptitle(title_str)

        def ani_init():
            poi_scat.set_array(env.episode_log_info_dict['poi_cur_value_array'][0] / self.env_conf['poi_value_max'])
            return poi_scat,

        def ani_update(frame):
            print('\r', frame, end='')
            poi_scat.set_array(env.episode_log_info_dict['poi_cur_value_array'][frame] / self.env_conf['poi_value_max'])
            # draw trace
            for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
                UGV_UAVs_Group = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id]
                # draw UGV trace
                ugv_id = UGV_UAVs_Group_id
                road_pos_list = UGV_UAVs_Group.ugv.episode_log_info_dict['final_road_pos']
                if frame > 0:
                    passed_road_node_id_list = \
                        UGV_UAVs_Group.ugv.episode_log_info_dict['final_passed_road_node_id_list'][-1][frame - 1]
                    ugv_coordx_list = []
                    ugv_coordy_list = []
                    ugv_coordx_list.append(env.road_pos2pos(road_pos_list[frame - 1])[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos_list[frame - 1])[1])
                    for passed_road_node_id in passed_road_node_id_list:
                        ugv_coordx_list.append(env.roads_net_dict[passed_road_node_id]['coordxy'][0])
                        ugv_coordy_list.append(env.roads_net_dict[passed_road_node_id]['coordxy'][1])
                    ugv_coordx_list.append(env.road_pos2pos(road_pos_list[frame])[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos_list[frame])[1])
                    plt.plot(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)], linewidth=2,
                             alpha=0.5)

                ugv_coordx_list = []
                ugv_coordy_list = []
                for road_pos in road_pos_list[frame:frame + 1]:
                    ugv_coordx_list.append(env.road_pos2pos(road_pos)[0])
                    ugv_coordy_list.append(env.road_pos2pos(road_pos)[1])
                plt.scatter(ugv_coordx_list, ugv_coordy_list, color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                            marker=self.marker_list[0], s=10)
                ugv_scat_list[ugv_id].set_offsets(np.array([[ugv_coordx_list[0], ugv_coordy_list[0] - 50]]))
                ugv_annotationbox_list[ugv_id].xybox = (ugv_coordx_list[0], ugv_coordy_list[0])

                # # draw collect range
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
                #     if step_id == frame:
                #         circ = plt.Circle((ugv_coordx_list[0], ugv_coordy_list[0]),
                #                           self.env_conf['uav_ugv_max_dis'] + self.env_conf['uav_sensing_range'],
                #                           color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                #                           alpha=0.2, fill=False)
                #         ax.add_patch(circ)
                #     if step_id > frame:
                #         break

                # draw uav trace
                if frame > 0:
                    for uav_id in range(self.env_conf['uav_num_each_group']):
                        uav = UGV_UAVs_Group.uav_list[uav_id]
                        pos_list = uav.episode_log_info_dict['final_pos']
                        uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                        uav_show_flag = False
                        uav_scat_id = str(ugv_id) + '-' + str(uav_id)
                        uav_annotationbox_dict[uav_scat_id].xybox = (pos_list[frame][0], pos_list[frame][1])
                        for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                            uav_sub_rollout_id].sub_episode_buffer_list:
                            uav_coordx_list = []
                            uav_coordy_list = []
                            if frame in sub_episode_buffer['step_id_s'] and frame - 1 in sub_episode_buffer[
                                'step_id_s']:
                                uav_coordx_list.append(pos_list[frame - 1][0])
                                uav_coordy_list.append(pos_list[frame - 1][1])
                                uav_coordx_list.append(pos_list[frame][0])
                                uav_coordy_list.append(pos_list[frame][1])
                                plt.plot(uav_coordx_list, uav_coordy_list,
                                         color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                                         linewidth=1,
                                         alpha=0.5)
                                plt.scatter(uav_coordx_list[-1], uav_coordy_list[-1],
                                            color=self.color_list[UGV_UAVs_Group_id % len(self.color_list)],
                                            marker=self.marker_list[(1 + uav_id) % len(self.marker_list)], s=0.5)

                                # uav_scat_dict[uav_scat_id].set_offsets(
                                #     np.array([[uav_coordx_list[-1], uav_coordy_list[-1] - 25]]))
                                uav_sensing_circle_dict[uav_scat_id].set_center(
                                    (uav_coordx_list[-1], uav_coordy_list[-1]))
                                uav_show_flag = True
                                break
                            if frame < sub_episode_buffer['step_id_s'][0]:
                                break
                        if not uav_show_flag:
                            # uav_scat_dict[uav_scat_id].set_offsets(np.array([[-100, -100]]))
                            uav_sensing_circle_dict[uav_scat_id].set_center((-100, -100))
            return [poi_scat] + ugv_scat_list + ugv_annotationbox_list

        ani = animation.FuncAnimation(
            fig=Fig,
            func=ani_update,
            frames=[step_id for step_id in range(self.env_conf['max_step_num'] + 1)],
            # frames=[step_id for step_id in range(10 + 1)],
            init_func=ani_init,
            interval=500,
            repeat=False,
            blit=True,
            save_count=self.env_conf['max_step_num'] + 1,
            # fargs=(3,)
        )

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
            video_dir_path = os.path.join(self.log_path, 'trace')
            if not os.path.exists(video_dir_path):
                os.makedirs(video_dir_path)
            video_path = os.path.join(video_dir_path, 'trace_iter_id' + str(iter_id) + '.mp4')
            ani.save(video_path)
        elif mode == 'test':
            video_dir_path = os.path.join(self.root_log_path, 'trace')
            if not os.path.exists(video_dir_path):
                os.makedirs(video_dir_path)
            # video_path = os.path.join(video_dir_path, 'subp_test_id' + str(subp_id) + '.mp4')
            video_path = os.path.join(video_dir_path, 'subp_test_id' + str(subp_id) + '.gif')
            # ani.save(video_path)
            ani.save(video_path, dpi=500)
        print('')
        plt.close()

    def draw_ana(self, iter_id, env, sub_rollout_manager):
        Fig = plt.figure()
        ax_dict = {}
        row_num = 5
        column_num = self.env_conf['UGV_UAVs_Group_num'] + 1
        sub_fig_id = 0

        eff = np.round(self.episode_metrics_result['eff'][iter_id], 2)
        fairness = np.round(self.episode_metrics_result['fairness'][iter_id], 2)
        dcr = np.round(self.episode_metrics_result['dcr'][iter_id], 2)
        hit = np.round(self.episode_metrics_result['hit'][iter_id], 2)
        ec = np.round(self.episode_metrics_result['ec'][iter_id], 2)
        ecr = np.round(self.episode_metrics_result['ecr'][iter_id], 2)

        title_str = 'iter: ' + str(iter_id) \
                    + ' eff: ' + str(eff) \
                    + ' fairness: ' + str(fairness) \
                    + ' dcr: ' + str(dcr) \
                    + ' ecr: ' + str(ecr) \
                    + '\n' \
                    + ' hit: ' + str(hit) \
                    + ' ec: ' + str(ec)

        # reward in episode
        ax_dict['reward'] = []
        for column_id in range(column_num):
            sub_fig_id += 1
            ax_dict['reward'].append(Fig.add_subplot(row_num, column_num, sub_fig_id))

        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            # draw UGV reward
            ugv_id = UGV_UAVs_Group_id
            ugv_sub_rollout_id = str(ugv_id)
            reward_array = np.zeros(self.env_conf['max_step_num'], dtype=np.float32)
            for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UGV'][
                ugv_sub_rollout_id].sub_episode_buffer_list:
                for reward_id, reward in enumerate(sub_episode_buffer['reward_s']):
                    reward_array[sub_episode_buffer['step_id_s'][reward_id]] = reward
            ax_dict['reward'][1 + UGV_UAVs_Group_id].plot(reward_array, c=self.color_list[0],
                                                          label='ugv_' + str(ugv_id))

            # draw UAV reward
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                reward_array = np.zeros(self.env_conf['max_step_num'], dtype=np.float32)
                for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                    uav_sub_rollout_id].sub_episode_buffer_list:
                    for reward_id, reward in enumerate(sub_episode_buffer['reward_s']):
                        reward_array[sub_episode_buffer['step_id_s'][reward_id]] = reward
                ax_dict['reward'][1 + UGV_UAVs_Group_id].plot(reward_array, c=self.color_list[(1 + uav_id) % len(self.color_list)],
                                                              label='uav_' + str(uav_id))
        # show uav act phases
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            ugv_id = UGV_UAVs_Group_id
            uav_id = 0
            uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
            for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                uav_sub_rollout_id].sub_episode_buffer_list:
                ax_dict['reward'][1 + UGV_UAVs_Group_id].fill_betweenx(np.linspace(0, 1, 3),
                                                                       sub_episode_buffer['step_id_s'][0],
                                                                       sub_episode_buffer['step_id_s'][-1], color='g')
            ax_dict['reward'][1 + UGV_UAVs_Group_id].set_title('reward for Group ' + str(UGV_UAVs_Group_id))
            ax_dict['reward'][1 + UGV_UAVs_Group_id].legend()

        # dcr in episode
        ax_dict['dcr'] = []
        for column_id in range(column_num):
            sub_fig_id += 1
            ax_dict['dcr'].append(Fig.add_subplot(row_num, column_num, sub_fig_id))

        dcr_group_array = np.zeros([self.env_conf['UGV_UAVs_Group_num'], self.env_conf['max_step_num']],
                                   dtype=np.float32)

        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            # draw UAV dcr
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                dcr_array = np.zeros(self.env_conf['max_step_num'], dtype=np.float32)
                for step_id in range(self.env_conf['max_step_num']):
                    dcr = np.sum(uav.episode_log_info_dict['final_data_collection'][step_id]) / np.sum(
                        env.poi_init_value_array)
                    dcr_array[step_id] = dcr
                dcr_group_array[UGV_UAVs_Group_id] += dcr_array
                ax_dict['dcr'][1 + UGV_UAVs_Group_id].plot(dcr_array, c=self.color_list[(1 + uav_id) % len(self.color_list)],
                                                           label='uav_' + str(uav_id))
            # draw group dcr
            ax_dict['dcr'][1 + UGV_UAVs_Group_id].plot(dcr_group_array[UGV_UAVs_Group_id], c=self.color_list[0],
                                                       label='group_' + str(UGV_UAVs_Group_id))
            ax_dict['dcr'][0].plot(dcr_group_array[UGV_UAVs_Group_id], c=self.color_list[(1 + UGV_UAVs_Group_id) % len(self.color_list)],
                                   label='group_' + str(UGV_UAVs_Group_id))
        ax_dict['dcr'][0].plot(np.sum(dcr_group_array, axis=0), c=self.color_list[0], label='all groups')
        ax_dict['dcr'][0].set_title('dcr for All Groups')
        ax_dict['dcr'][0].legend()

        # show uav act phases
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            ugv_id = UGV_UAVs_Group_id
            uav_id = 0
            uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
            for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                uav_sub_rollout_id].sub_episode_buffer_list:
                ax_dict['dcr'][1 + UGV_UAVs_Group_id].fill_betweenx(np.linspace(0, 1, 3),
                                                                    sub_episode_buffer['step_id_s'][0],
                                                                    sub_episode_buffer['step_id_s'][-1], color='g')
            ax_dict['dcr'][1 + UGV_UAVs_Group_id].set_title('dcr for Group ' + str(UGV_UAVs_Group_id))
            ax_dict['dcr'][1 + UGV_UAVs_Group_id].legend()

        # energy in episode
        ax_dict['energy'] = []
        for column_id in range(column_num):
            sub_fig_id += 1
            ax_dict['energy'].append(Fig.add_subplot(row_num, column_num, sub_fig_id))

        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            # draw UAV energy
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                energy_array = np.array(uav.episode_log_info_dict['final_energy'])
                ax_dict['energy'][1 + UGV_UAVs_Group_id].plot(energy_array, c=self.color_list[(1 + uav_id) % len(self.color_list)],
                                                              label='uav_' + str(uav_id))

        # show uav act phases
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            ugv_id = UGV_UAVs_Group_id
            uav_id = 0
            uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
            for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                uav_sub_rollout_id].sub_episode_buffer_list:
                ax_dict['energy'][1 + UGV_UAVs_Group_id].fill_betweenx(np.linspace(0, 1, 3),
                                                                       sub_episode_buffer['step_id_s'][0],
                                                                       sub_episode_buffer['step_id_s'][-1], color='g')
            ax_dict['energy'][1 + UGV_UAVs_Group_id].set_title('energy for Group ' + str(UGV_UAVs_Group_id))
            ax_dict['energy'][1 + UGV_UAVs_Group_id].legend()
        # hit in episode
        ax_dict['hit'] = []
        for column_id in range(column_num):
            sub_fig_id += 1
            ax_dict['hit'].append(Fig.add_subplot(row_num, column_num, sub_fig_id))

        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            # draw UAV hit
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                hit_array = np.array(uav.episode_log_info_dict['final_hit'])
                ax_dict['hit'][1 + UGV_UAVs_Group_id].plot(hit_array, c=self.color_list[(1 + uav_id) % len(self.color_list)],
                                                           label='uav_' + str(uav_id))

        # show uav act phases
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            ugv_id = UGV_UAVs_Group_id
            uav_id = 0
            uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
            for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                uav_sub_rollout_id].sub_episode_buffer_list:
                ax_dict['hit'][1 + UGV_UAVs_Group_id].fill_betweenx(np.linspace(0, 1, 3),
                                                                    sub_episode_buffer['step_id_s'][0],
                                                                    sub_episode_buffer['step_id_s'][-1], color='g')
            ax_dict['hit'][1 + UGV_UAVs_Group_id].set_title('hit for Group ' + str(UGV_UAVs_Group_id))
            ax_dict['hit'][1 + UGV_UAVs_Group_id].legend()

        # out_of_ugv in episode
        ax_dict['out_of_ugv'] = []
        for column_id in range(column_num):
            sub_fig_id += 1
            ax_dict['out_of_ugv'].append(Fig.add_subplot(row_num, column_num, sub_fig_id))

        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            # draw UAV out_of_ugv
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                out_of_ugv_array = np.array(uav.episode_log_info_dict['final_out_of_ugv'])
                ax_dict['out_of_ugv'][1 + UGV_UAVs_Group_id].plot(out_of_ugv_array, c=self.color_list[(1 + uav_id) % len(self.color_list)],
                                                                  label='uav_' + str(uav_id))

        # show uav act phases
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            ugv_id = UGV_UAVs_Group_id
            uav_id = 0
            uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
            for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                uav_sub_rollout_id].sub_episode_buffer_list:
                ax_dict['out_of_ugv'][1 + UGV_UAVs_Group_id].fill_betweenx(np.linspace(0, 1, 3),
                                                                           sub_episode_buffer['step_id_s'][0],
                                                                           sub_episode_buffer['step_id_s'][-1],
                                                                           color='g')
            ax_dict['out_of_ugv'][1 + UGV_UAVs_Group_id].set_title('out_of_ugv for Group ' + str(UGV_UAVs_Group_id))
            ax_dict['out_of_ugv'][1 + UGV_UAVs_Group_id].legend()

        plt.tight_layout()
        plt.suptitle(title_str)
        fig_dir_path = os.path.join(self.log_path, 'ana')
        if not os.path.exists(fig_dir_path):
            os.makedirs(fig_dir_path)
        fig_path = os.path.join(fig_dir_path, 'ana_train_step_' + str(iter_id) + '.png')
        # Fig.savefig(fig_path)
        Fig.savefig(fig_path, dpi=1300)
        plt.close()

    def visual_obs(self, sub_iter_counter, step_id, obs, obs_name):
        if 'list' in obs_name:
            channel_num = obs[0].shape[0]
            obs_size = obs[0].shape[1]
            obs_num = len(obs)
            total_obs = np.zeros([obs_size * channel_num + channel_num - 1, obs_size * obs_num + obs_num - 1],
                                 dtype=np.float32)
            for obs_id in range(obs_num):
                for channel_id in range(channel_num):
                    total_obs[channel_id * (obs_size + 1):(channel_id + 1) * (obs_size + 1) - 1,
                    obs_id * (obs_size + 1):(obs_id + 1) * (obs_size + 1) - 1] = obs[obs_id][channel_id]
        elif len(obs.shape) == 3:
            channel_num = obs.shape[0]
            obs_size = obs.shape[1]
            total_obs = np.zeros([obs_size * channel_num + channel_num - 1, obs_size], dtype=np.float32)
            for channel_id in range(channel_num):
                total_obs[channel_id * (obs_size + 1):(channel_id + 1) * (obs_size + 1) - 1, :] = obs[channel_id]
        elif len(obs.shape) == 2:
            total_obs = obs
        image_dir_path = os.path.join(self.log_path, 'visual_obs', 'iter_' + str(sub_iter_counter), obs_name)
        if not os.path.exists(image_dir_path):
            os.makedirs(image_dir_path)
        image_path = os.path.join(image_dir_path, 'obs_name_' + str(step_id) + '.png')
        total_obs_T = total_obs.T
        total_obs_T_convert = np.zeros_like(total_obs_T, dtype=np.float32)
        total_obs_T_convert[[i for i in range(total_obs.shape[1])]] = total_obs_T[
            [i for i in reversed(range(total_obs.shape[1]))]]
        imageio.imsave(image_path, total_obs_T_convert)
