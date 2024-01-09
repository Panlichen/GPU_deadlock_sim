import pprint
import random
from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
import json
import argparse
from multiprocessing import Process, Value, Lock, current_process

from detect_ring import RingDetector

# constant str
SINGLE_MODEL = "Single"
STREAM_WITH_SYNC_MODEL = "StreamWithSync"
MEGATRON_GROUPING_POLICY = "Megatron"
ARBITRARY_GROUPING_POLICY = "Arbitrary"

# print log control
PRINT_PARSE_RAW = True
PRINT_GROUP_CLASS = False
PRINT_GPU_CLASS = False
PRINT_MAIN_LOOP = True
PRINT_GRAPH = False
PRINT_DEADLOCK_DETAIL = False
PRINT_ROUND_REPORT = True
PRINT_PARSE_AT_BEGIN = True


class GPU(ABC):
    def __init__(self, gpu_id, group_of_gpu, expect_coll_exec_list, coll_2_group, disorder_prob):
        self.gpu_id = gpu_id
        self.group_of_gpu = group_of_gpu
        self.expect_coll_exec_list = expect_coll_exec_list
        self.coll_num = len(expect_coll_exec_list)
        self.disorder_prob = disorder_prob
        self.coll_2_group = coll_2_group

        self.actual_coll_exec_list = []
        self.is_hang = False
        self.submit_counter = 0

        self.reset()
    
    def __str__(self):
        attributes = ", ".join(f"{key}={value}"
                               if key != 'group_of_gpu' and key != 'coll_2_group' else ""
                               for key, value in self.__dict__.items())
        group_info = ", ".join(f"{group.group_id}" for group in self.group_of_gpu)
        attributes += f", groups: ({group_info})"

        coll_2_group_str = ", ".join(f"{coll_id} -> {self.coll_2_group[coll_id].group_id}" for coll_id in self.coll_2_group)
        attributes += f", coll_2_group: ({coll_2_group_str})"

        return f"{self.__class__.__name__}({attributes})"

    def reset(self):
        self.is_hang = False
        self.submit_counter = 0
        self.actual_coll_exec_list = self.expect_coll_exec_list.copy()

    def decide_coll(self, round_id) -> int:  # 为了适应single的需求，改为返回决定的coll_id
               
        # 决定要执行的coll
        decided_coll = -1
        index_in_actual_list = 0
        while not decided_coll >= 0:

            random_number = random.random()  # Generate a random number between 0 and 1
            assert index_in_actual_list < len(self.actual_coll_exec_list), f"index_in_actual_list = {index_in_actual_list}, len(self.actual_coll_exec_list) = {len(self.actual_coll_exec_list)}"
            if index_in_actual_list == len(self.actual_coll_exec_list) -1 or random_number >= self.disorder_prob:
                # 如果当前已经是最后一个了，或者决定不乱序，选当前coll
                decided_coll = self.actual_coll_exec_list[index_in_actual_list]
                self.submit_counter += 1
            else:
                # 否则决定乱序，不选当前的
                index_in_actual_list += 1

        # 拿到要执行的coll后，更新actual_coll_exec_list
        self.actual_coll_exec_list = self.actual_coll_exec_list[:index_in_actual_list] + self.actual_coll_exec_list[index_in_actual_list+1:]

        return decided_coll
              
    @abstractmethod
    def step_forward(self, round_id) -> int:
        pass

class SingleQueueGPU(GPU):
    def __init__(self, gpu_id, group_of_gpu, expect_coll_exec_list, coll_2_group, disorder_prob):
        super().__init__(gpu_id, group_of_gpu, expect_coll_exec_list, coll_2_group, disorder_prob)

    def step_forward(self, round_id) -> int:
        # bugfix: 这里不能直接发，需要和group交互一下，看看自己能不能发，能发的前提是自己刚刚提交的coll已经执行了，执行的前提是所有这个group的GPU都提交了。
        # 所以在group里要设计一个list，标记已经提交的GPU，这个GPU可以提交的条件是，自己不在group的list中。
        # 这还比较麻烦。。。因为这时候GPU还不知道自己要提交哪个coll，也就不知道自己要向那个group询问自己能不能提交。
        
        decided_coll = self.decide_coll(round_id)

        if self.coll_2_group[decided_coll].gpu_can_submit(self.gpu_id):
            # 向group提交coll，返回值：-1死锁，0成功。提交coll不会有hang的情况。
            ret = self.coll_2_group[decided_coll].submit(self.gpu_id, decided_coll, round_id)
            assert ret != 1, "提交coll不会导致hang"
            
            return ret


class StreamWithSyncGPU(GPU):
    def __init__(self, gpu_id, group_of_gpu, expect_coll_exec_list, coll_2_group, disorder_prob, sync_prob):
        super().__init__(gpu_id, group_of_gpu, expect_coll_exec_list, coll_2_group, disorder_prob)

        self.sync_prob = sync_prob

    def step_forward(self, round_id) -> int:
        random_number = random.random()
        if random_number < self.sync_prob:
            final_ret = 0
            for group in self.group_of_gpu:
                ret= group.submit(self.gpu_id, -1, round_id)
                if ret == 1:  # 向所有group提交sync操作，submit改变状态，所以需要都循环完，在每个group视角下都处理一遍。
                    self.is_hang = True
                    final_ret = 1
                elif ret == -1:
                    return -1  # 如果有死锁，直接返回，不用再循环了。
            return final_ret
        else:
            decided_coll = self.decide_coll(round_id)
        
            # 向group提交coll，返回值：-1死锁，0成功。提交coll不会有hang的情况。
            ret = self.coll_2_group[decided_coll].submit(self.gpu_id, decided_coll, round_id)
            assert ret != 1, "提交coll不会导致hang"
            
            return ret

    def gpu_check_hang(self) -> bool:
        self.is_hang = False
        for group in self.group_of_gpu:
            if group.group_check_hang(self.gpu_id):  # group_check_hang不改变状态，所以可以不全循环完，碰到一个依然hang的就返回。
                self.is_hang = True
                break
        return self.is_hang


class Group(ABC):
    def __init__(self, group_id, gpus, expected_colls):
        self.group_id = group_id
        self.gpu_cnt = len(gpus)
        # 下边这两个使用浅复制就好，并不会被修改。
        self.gpus = gpus
        self.expected_colls = expected_colls

        self.coll_submit_counter = {}

    def __str__(self):
        attributes = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def submit(self, gpu_id, coll_id, round_id) -> int:
        pass


class SingleQueueGroup(Group):
    def __init__(self, group_id, gpus, expected_colls):
        super().__init__(group_id, gpus, expected_colls)
        self.submitted_undone_colls = []
        self.current_submitted_gpus = []

        self.reset()

    def reset(self):
        self.submitted_undone_colls = []
        self.current_submitted_gpus = []

    def gpu_can_submit(self, gpu_id) -> bool:
        return not gpu_id in self.current_submitted_gpus

    def submit(self, gpu_id, coll_id, round_id) -> int:
        # 两个主要工作：检查死锁，更新状态
        self.submitted_undone_colls.append(coll_id)
        self.current_submitted_gpus.append(gpu_id)
        
        if PRINT_MAIN_LOOP:
            print(f"Round {round_id}, Group {self.group_id} GPU {gpu_id} submit coll {coll_id}, self.submitted_undone_colls: {self.submitted_undone_colls}", flush=True)

        if len(set(self.submitted_undone_colls)) > 1:  # 并不是所有已提交的coll都相等，死锁，而且不需要等到所有GPU都提交了。
            if PRINT_DEADLOCK_DETAIL:
                self.print_deadlock_info(gpu_id, round_id)
            return -1

        if len(self.submitted_undone_colls) == len(self.gpus):
            # 如果所有GPU都提交了，还没有死锁，那么就可以清空了。
            self.submitted_undone_colls = []
            self.current_submitted_gpus = []
        
        return 0

    def print_deadlock_info(self, gpu_id, round_id):
        print(f"!!SingleQueue Deadlock!! Round {round_id}, Group {self.group_id} GPU {gpu_id}, {len(self.submitted_undone_colls)} / {len(self.gpus)} current submitted_undone_colls: {self.submitted_undone_colls}", flush=True)


class StreamWithSyncGroup(Group):
    def __init__(self, group_id, gpus, expected_colls, resource_limit):
        super().__init__(group_id, gpus, expected_colls)
        self.submitted_undone_colls_in_group = set()
        self.unsubmitted_colls_from_gpu = {}
        self.submitted_undone_colls_from_gpu = {}
        self.hang_gpus = set()
        self.resource_limit = resource_limit

        self.reset()

    def reset(self):
        self.submitted_undone_colls_in_group = set()
        self.hang_gpus = set()
        for gpu_id in self.gpus:
            self.unsubmitted_colls_from_gpu[gpu_id] = set(self.expected_colls)
            self.submitted_undone_colls_from_gpu[gpu_id] = set()
        
        for coll_id in self.expected_colls:
            self.coll_submit_counter[coll_id] = self.gpu_cnt

    def group_check_hang(self, gpu_id) -> bool:
        gpu_submitted_undone_colls = self.submitted_undone_colls_from_gpu[gpu_id]
        ret = len(gpu_submitted_undone_colls.
                   intersection(self.submitted_undone_colls_in_group)) > 0
        if ret and PRINT_MAIN_LOOP:
            print(f"Group {self.group_id} GPU {gpu_id} hang, gpu_submitted_undone_colls: {gpu_submitted_undone_colls}, submitted_undone_colls_in_group: {self.submitted_undone_colls_in_group}", flush=True)
        return ret

    def submit(self, gpu_id, coll_id, round_id) -> int:
        assert not gpu_id in self.hang_gpus, "已经hang的gpu不能提交任何东西"

        ret = 0

        if coll_id >= 0:  # 提交普通coll
            assert coll_id in self.expected_colls, "提交了错误的coll"

            self.coll_submit_counter[coll_id] -= 1

            if PRINT_MAIN_LOOP:
                print(f"Round {round_id}, Group {self.group_id} GPU {gpu_id} submit coll {coll_id}, submit_counter: {self.gpu_cnt - self.coll_submit_counter[coll_id]} / {self.gpu_cnt}", flush=True)
            self.submitted_undone_colls_from_gpu[gpu_id].add(coll_id)
            self.unsubmitted_colls_from_gpu[gpu_id].remove(coll_id)

            if self.coll_submit_counter[coll_id] > 0:
                self.submitted_undone_colls_in_group.add(coll_id)
            else:
                self.submitted_undone_colls_in_group.remove(coll_id)
                for gpu_id in self.gpus:
                    self.submitted_undone_colls_from_gpu[gpu_id].remove(coll_id)
            
            if self.resource_limit > 0:
                all_full_deadlock = True
                for check_gpu_id in self.gpus:
                    if len(self.submitted_undone_colls_from_gpu[check_gpu_id]) < self.resource_limit:
                        all_full_deadlock = False
                        break
                if all_full_deadlock:
                    print(f"all gpus exceed resource limit {self.resource_limit}", flush=True)
                    return -1

            new_hang_gpus = set()
            for hang_gpu_id in self.hang_gpus:
                if self.group_check_hang(hang_gpu_id):
                    new_hang_gpus.add(hang_gpu_id)
            self.hang_gpus = new_hang_gpus
            
            if PRINT_MAIN_LOOP:
                print(f"after submit normal coll, Round {round_id}, Group {self.group_id} GPU {gpu_id} submit coll {coll_id}, group hang_gpus: {self.hang_gpus}, group submitted_undone_colls_in_group: {self.submitted_undone_colls_in_group}, gpu submitted_undone_colls_from_gpu: {self.submitted_undone_colls_from_gpu[gpu_id]}, gpu unsubmitted_colls_from_gpu: {self.unsubmitted_colls_from_gpu[gpu_id]}", flush=True)

        else:  # 提交sync
            if PRINT_MAIN_LOOP:
                print(f"Round {round_id}, Group {self.group_id} GPU {gpu_id} submit sync {coll_id}", flush=True)
            if self.group_check_hang(gpu_id):  # 判断hang
                ret = 1
                self.hang_gpus.add(gpu_id)

            if len(self.hang_gpus) >= 2:  # bug here. 之前写成了>2，导致死锁检测不出来。
                # 判断死锁：
                # 建图：
                coll_graph = {}
                for hang_gpu_id in self.hang_gpus:
                    for coll_id in self.submitted_undone_colls_from_gpu[hang_gpu_id]:
                        # submitted_undone_colls的边都指向其他GPU的unsubmitted_colls中和自己同名的coll
                        coll_node = "-".join([str(self.group_id), str(hang_gpu_id), str(coll_id)]) + "-S"
                        coll_graph.setdefault(coll_node, [])
                        for other_hang_gpu_id in self.hang_gpus:
                            if other_hang_gpu_id == hang_gpu_id:
                                continue
                            for unsubmitted_coll_id in self.unsubmitted_colls_from_gpu[other_hang_gpu_id]:
                                if unsubmitted_coll_id == coll_id:
                                    unsubmitted_coll_node = "-".join([str(self.group_id), str(other_hang_gpu_id), str(unsubmitted_coll_id)]) + "-U"
                                    coll_graph[coll_node].append(unsubmitted_coll_node)
                        
                    for coll_id in self.unsubmitted_colls_from_gpu[hang_gpu_id]:
                        # unsubmitted_colls的边都指向同一个GPU中所有submitted_undone_colls中的coll
                        coll_node = "-".join([str(self.group_id), str(hang_gpu_id), str(coll_id)]) + "-U"
                        coll_graph.setdefault(coll_node, [])
                        # for submitted_undone_coll_id in self.submitted_undone_colls_from_gpu[hang_gpu_id]:
                        #     submitted_undone_coll_node = "-".join([str(self.group_id), str(hang_gpu_id), str(submitted_undone_coll_id)])
                        #     coll_graph[coll_node].append(submitted_undone_coll_node)
                        coll_graph[coll_node].extend(["-".join([str(self.group_id), str(hang_gpu_id), str(submitted_undone_coll_id)]) + "-S" for submitted_undone_coll_id in self.submitted_undone_colls_from_gpu[hang_gpu_id]])
                if PRINT_GRAPH:
                    self.print_graph(round_id, coll_graph)

                # 检测环
                
                if PRINT_MAIN_LOOP:
                    print(f"Round {round_id}, Group {self.group_id} GPU {gpu_id} before detect_ring", flush=True) 
                detector = RingDetector()  # bug here。最一开始没有用class，vis、trace数组是全局变量，导致数组没有清空，每次数组内容都累加
                result = detector.detect_ring(coll_graph)
                if PRINT_MAIN_LOOP:
                    print(f"Round {round_id}, Group {self.group_id} GPU {gpu_id} detect_ring return {result}", flush=True) 
                if result:
                    ret = -1
                    if PRINT_DEADLOCK_DETAIL:
                        self.print_deadlock_info(round_id, coll_graph)

        return ret
            
    def print_graph(self, round_id, coll_graph):
        for hang_gpu_id in self.hang_gpus:
            print(f"print_graph hang gpu {hang_gpu_id} has submitted undone colls: {self.submitted_undone_colls_from_gpu[hang_gpu_id]}", flush=True)
            print(f"print_graph hang gpu {hang_gpu_id} has unsubmitted colls: {self.unsubmitted_colls_from_gpu[hang_gpu_id]}", flush=True)
        print(f"Round {round_id}, Group {self.group_id} coll_graph: ", flush=True)
        pprint.pprint(coll_graph)
        

    def print_deadlock_info(self, round_id, coll_graph):
        print(f"!!StreamWithSync Deadlock!! Round {round_id}, hang gpus: {self.hang_gpus}", flush=True)
        for hang_gpu_id in self.hang_gpus:
            print(f"hang gpu {hang_gpu_id} has submitted undone colls: {self.submitted_undone_colls_from_gpu[hang_gpu_id]}", flush=True)
            print(f"hang gpu {hang_gpu_id} has unsubmitted colls: {self.unsubmitted_colls_from_gpu[hang_gpu_id]}", flush=True)
        print(f"coll_graph: {coll_graph}", flush=True)


def parse_config(config):

    fix_seed = config['fix_seed']
    if fix_seed == 1:
        random.seed(2023)  # 42：这是一个在科技和编程文化中非常流行的数字，源自道格拉斯·亚当斯的科幻小说《银河系漫游指南》，其中它被描绘为“生命、宇宙以及任何事物的终极答案”。

    total_rounds = config['total_rounds']
    disorder_prob = config['disorder_prob']
    sync_prob = config['sync_prob']
    model = config['model']
    grouping_policy = config['grouping_policy']

    if grouping_policy == MEGATRON_GROUPING_POLICY:
        tp_group_size = config['tp_group_size']
        dp_group_size = config['dp_group_size']
        pp_group_size = config['pp_group_size']

        gpu_num = tp_group_size * dp_group_size * pp_group_size

        group_num = (tp_group_size + dp_group_size) * pp_group_size

        coll_cnt_per_tp_group = config['coll_cnt_per_tp_group']
        coll_cnt_per_dp_group = config['coll_cnt_per_dp_group']
        # 这里是一个简化的实现。
        # 一个PP组内有dp_group_size个TP组，各个TP组对模型的切分方法是一样的，==所以各个TP组的coll数目是一致的==。
        # 一个PP组内有tp_group_size个DP组，各个DP组是对模型不同部分的参数进行梯度聚合，==所以各个DP组的coll数目可能不一致==
        # 不同PP组之间，TP组的coll数目会不同，DP组coll数目更会不同
        # 完整的配置应该是长度为pp_group_size的列表，里边包含一个字典：
            # coll_cnt_per_tp_group对应一个数字
            # coll_cnt_per_dp_group是一个长度为tp_group_size的列表
        
        if PRINT_PARSE_RAW:
            print(f"MEGATRON_GROUPING_POLICY: tp_group_size = {tp_group_size}, dp_group_size = {dp_group_size}, pp_group_size = {pp_group_size}, coll_cnt_per_tp_group = {coll_cnt_per_tp_group}, coll_cnt_per_dp_group = {coll_cnt_per_dp_group}", flush=True)

        gpu_per_group = []
        coll_per_group = []
        curr_gpu_id = 0
        curr_coll_id = 0
        # 分配tp group、dp group的gpu和coll。一个pp group里有tp_group_size * dp_group_size个GPU，dp_group_size(即tp group的个数) * coll_cnt_per_tp_group + tp_group_size(即dp group的个数) * coll_cnt_per_dp_group 个coll。
        for pp_group_id in range(pp_group_size):
            if PRINT_PARSE_RAW:
                print(f"pp group {pp_group_id}", flush=True)
            tp_groups = []

            for tp_group_id in range(dp_group_size):  # dp_group_size(即tp group的个数)
                group_id = (dp_group_size + tp_group_size) * pp_group_id + tp_group_id
                # 分配GPU
                new_tp_group = [curr_gpu_id + i for i in range(tp_group_size)]
                gpu_per_group.append(new_tp_group)
                curr_gpu_id += tp_group_size
                tp_groups.append(new_tp_group)
                
                # 分配coll
                coll_per_group.append([curr_coll_id + i for i in range(coll_cnt_per_tp_group)])
                curr_coll_id += coll_cnt_per_tp_group

                if PRINT_PARSE_RAW:
                    print(f"TP group id: {group_id}, has gpus: {gpu_per_group[group_id]}, colls: {coll_per_group[group_id]}", flush=True)
                
            dp_groups = list(map(list, zip(*tp_groups)))  # 实现了矩阵转置

            for dp_group_id in range(tp_group_size):  # tp_group_size(即dp group的个数)
                group_id = (dp_group_size + tp_group_size) * pp_group_id + dp_group_size + dp_group_id
                # 分配GPU，不分配新GPU了，而是从前边分配的TP group里边取，同一个pp group里的所有tp group的第dp_group_id个gpu组成dp group。
                gpu_per_group.append(dp_groups[dp_group_id])

                # 分配coll
                coll_per_group.append([curr_coll_id + i for i in range(coll_cnt_per_dp_group)])
                curr_coll_id += coll_cnt_per_dp_group
            
                if PRINT_PARSE_RAW:
                    print(f"DP group id: {group_id}, has gpus: {gpu_per_group[group_id]}, colls: {coll_per_group[group_id]}", flush=True)

    elif grouping_policy == ARBITRARY_GROUPING_POLICY:
        gpu_num = config['gpu_num']
        group_num = config['group_num']
        gpu_per_group = config['gpu_per_group']

        assert len(gpu_per_group) == group_num, "Number of gpu_per_group mismatch!"

        gpu_union = set()
        max_gpu_id = -1
        for gpu_list in gpu_per_group:
            gpu_union = gpu_union.union(set(gpu_list))
            max_gpu_id = max_gpu_id if max_gpu_id >= max(gpu_list) else max(gpu_list)
        if PRINT_PARSE_RAW:
            print(f"gpu_union: {gpu_union}, gpu_num: {gpu_num}, max_gpu_id: {max_gpu_id}", flush=True)
        assert len(gpu_union) == gpu_num, f"GPU数目与分组不匹配，gpu_union: {gpu_union}, len(gpu_union): {len(gpu_union)}, gpu_num: {gpu_num}, max_gpu_id: {max_gpu_id}"
        assert max_gpu_id == gpu_num - 1, "GPU ID 分配不合理"

        coll_cnt_per_group = config['coll_cnt_per_group']
        if PRINT_PARSE_RAW:
            print(f"ARBITRARY_GROUPING_POLICY: coll_cnt_per_group = {coll_cnt_per_group}", flush=True)
        assert len(coll_cnt_per_group) == group_num, "Coll allocation and Number of gpu_per_group mismatch!"
        coll_per_group = []
        curr_coll_id = 0
        for group_id in range(group_num):
            coll_per_group.append([curr_coll_id + i for i in range(coll_cnt_per_group[group_id])])
            curr_coll_id += coll_cnt_per_group[group_id]
            if PRINT_PARSE_RAW:
                print(f"Group id: {group_id}, has gpus: {gpu_per_group[group_id]}, colls: {coll_per_group[group_id]}", flush=True)

    coll_num = sum(len(row) for row in coll_per_group)

    # 建立从coll到group的倒排索引，在这里统一做的好处是不同的group的coll不会重叠。
    coll_2_group_id = {}
    for group_id in range(group_num):
        for coll_id in coll_per_group[group_id]:
            coll_2_group_id[coll_id] = group_id

    return total_rounds, gpu_num, group_num, coll_num, disorder_prob, sync_prob, model, grouping_policy, gpu_per_group, coll_per_group, coll_2_group_id


def dispatch_coll_2_gpu(gpu_per_group, coll_per_group):

    assert len(gpu_per_group) == len(coll_per_group), "这咋能不一样"

    #===================================================
    # coll_per_gpu = {}
    
    # for group_id in range(len(gpu_per_group)):
    #     gpu_list = gpu_per_group[group_id]
    #     coll_list = coll_per_group[group_id]
    #     for gpu_id in gpu_list:
    #         coll_per_gpu.setdefault(gpu_id, []).extend(coll_list)
    #===================================================

    #===================================================
    coll_per_gpu_2d = {}
    max_sublist_len_per_gpu = {}
    coll_per_gpu = {}
    for group_id in range(len(gpu_per_group)):
        gpu_list = gpu_per_group[group_id]
        coll_list = coll_per_group[group_id]
        for gpu_id in gpu_list:
            coll_per_gpu_2d.setdefault(gpu_id, []).append(coll_list)
            if gpu_id in max_sublist_len_per_gpu:
                max_sublist_len_per_gpu[gpu_id] = max(max_sublist_len_per_gpu[gpu_id], len(coll_list))
            else:
                max_sublist_len_per_gpu[gpu_id] = len(coll_list)

    for gpu_id in coll_per_gpu_2d:
        coll_per_gpu.setdefault(gpu_id, [])
        for index in range(max_sublist_len_per_gpu[gpu_id]):
            for sub_coll_list in coll_per_gpu_2d[gpu_id]:
                if index < len(sub_coll_list):
                    coll_per_gpu[gpu_id].append(sub_coll_list[index])
    #===================================================

    if PRINT_PARSE_RAW:
        for gpu_id in coll_per_gpu:
            print(f"gpu {gpu_id} has colls: {coll_per_gpu[gpu_id]}", flush=True)
    
    return coll_per_gpu
        

def get_groups_of_gpus(gpu_per_group):
    group_ids_of_gpus = {}
    for group_id in range(len(gpu_per_group)):
        gpu_list = gpu_per_group[group_id]
        for gpu_id in gpu_list:
            group_ids_of_gpus.setdefault(gpu_id, []).append(group_id)
    return group_ids_of_gpus


def all_gpu_submit_done(gpus):
    for gpu in gpus:
        if gpu.submit_counter < gpu.coll_num:
            if PRINT_MAIN_LOOP:
                print(f"GPU {gpu.gpu_id} submit_counter: {gpu.submit_counter} < coll_num: {gpu.coll_num}", flush=True)
            return False
    return True


def main_loop(gpus, groups, total_rounds, sum_deadlock_rounds, lock):
    deadlock_counter = 0
    for round_id in range(total_rounds):
        if PRINT_MAIN_LOOP:
            print(f"Round {round_id}", flush=True)
        for gpu in gpus:
            gpu.reset()
        for group in groups: 
            group.reset()

        # 判定一个round结束，要么发生死锁，要么所有GPU都提交完了。
        # 提交完的标准是 gpu.submit_counter == gpu.coll_num
        while not all_gpu_submit_done(gpus):
            # 每个GPU提交一个coll
            encounter_deadlock = False
            for gpu in gpus:
                # if PRINT_MAIN_LOOP:
                #     print(f"Round {round_id} GPU {gpu.gpu_id} works", flush=True)
                if gpu.is_hang:
                    assert isinstance(gpu, StreamWithSyncGPU), "hang的GPU必须是StreamWithSyncGPU"
                    if gpu.gpu_check_hang():
                        # if PRINT_MAIN_LOOP:
                        #     print(f"Round {round_id} GPU {gpu.gpu_id} is hang, skip", flush=True)
                        continue
                if gpu.submit_counter == gpu.coll_num: # 这个GPU已经提交完了，跳过
                    # if PRINT_MAIN_LOOP:
                    #     print(f"Round {round_id} GPU {gpu.gpu_id} submit_counter: {gpu.submit_counter} == coll_num: {gpu.coll_num}, skip", flush=True)
                    continue
                if PRINT_MAIN_LOOP:
                    print(f"Round {round_id} GPU {gpu.gpu_id} submits", flush=True)
                ret = gpu.step_forward(round_id)
                if ret == -1:
                    deadlock_counter += 1
                    encounter_deadlock = True
                    if PRINT_DEADLOCK_DETAIL:
                        print(f"Deadlock! Round {round_id}, GPU {gpu.gpu_id}", flush=True)
                    break
            if encounter_deadlock:
                break
        
        if PRINT_ROUND_REPORT:
            print(f"currently #Deadlock round: {deadlock_counter} out of {round_id + 1} rounds, ratio: {deadlock_counter / (round_id + 1)}\n\n", flush=True)
    
    with lock:
        sum_deadlock_rounds.value += deadlock_counter
    proc = current_process()

    print(f"\n\n~~~Process {proc.pid} #Deadlock round: {deadlock_counter} out of {total_rounds} rounds, ratio: {deadlock_counter / total_rounds}", flush=True)


def init_7_main_loop(config, sum_deadlock_rounds, lock):

    total_rounds, gpu_num, group_num, coll_num, disorder_prob, sync_prob, model, grouping_policy, gpu_per_group, coll_per_group, coll_2_group_id = parse_config(config)

    if "resource_limit" in config:
        resource_limit = config['resource_limit']
    else:
        resource_limit = -1
    
    proc = current_process()

    if PRINT_PARSE_AT_BEGIN:
        print(f"Process {proc.pid} model: {model}, gropuing_policy: {grouping_policy} gpu_num: {gpu_num}, group_num: {group_num}, coll_num: {coll_num}, disorder_prob: {disorder_prob}, sync_prob: {sync_prob}, resource_limit: {resource_limit}", flush=True)
        if grouping_policy == MEGATRON_GROUPING_POLICY:
            print(f"tp_group_size: {config['tp_group_size']}, dp_group_size: {config['dp_group_size']}, pp_group_size: {config['pp_group_size']}, coll_cnt_per_tp_group: {config['coll_cnt_per_tp_group']}, coll_cnt_per_dp_group: {config['coll_cnt_per_dp_group']}", flush=True)

    coll_per_gpu = dispatch_coll_2_gpu(gpu_per_group, coll_per_group)

    group_ids_of_gpus = get_groups_of_gpus(gpu_per_group)

    # 初始化Group，主要利用gpu_per_group和coll_per_group
    if model == SINGLE_MODEL:
        groups = [SingleQueueGroup(group_id, gpu_per_group[group_id], coll_per_group[group_id]) for group_id in range(group_num)]
    elif model == STREAM_WITH_SYNC_MODEL:
        groups = [StreamWithSyncGroup(group_id, gpu_per_group[group_id], coll_per_group[group_id], resource_limit) for group_id in range(group_num)]

    if PRINT_GROUP_CLASS:
        for group in groups:
            print(group, flush=True)

    coll_2_group = {}
    for coll_id in coll_2_group_id:
        coll_2_group[coll_id] = groups[coll_2_group_id[coll_id]]
    
    # 初始化GPU，主要利用coll_per_gpu和groups
    if model == SINGLE_MODEL:
        gpus = [SingleQueueGPU(
            gpu_id,
            [groups[group_id] for group_id in group_ids_of_gpus[gpu_id]],
            coll_per_gpu[gpu_id],
            coll_2_group,
            disorder_prob,
        ) for gpu_id in range(gpu_num)]
    elif model == STREAM_WITH_SYNC_MODEL:
        gpus = [StreamWithSyncGPU(
            gpu_id,
            [groups[group_id] for group_id in group_ids_of_gpus[gpu_id]],
            coll_per_gpu[gpu_id],
            coll_2_group,
            disorder_prob,
            sync_prob,
        ) for gpu_id in range(gpu_num)]

    if PRINT_GPU_CLASS:
        for gpu in gpus:
            print(gpu, flush=True)

    print(f"Process {proc.pid} before main loop", flush=True)

    main_loop(gpus, groups, total_rounds, sum_deadlock_rounds, lock)
    
    print(f"Process {proc.pid} model: {model}, gropuing_policy: {grouping_policy} gpu_num: {gpu_num}, group_num: {group_num}, coll_num: {coll_num}, disorder_prob: {disorder_prob}, sync_prob: {sync_prob}, resource_limit: {resource_limit}", flush=True)
    if grouping_policy == MEGATRON_GROUPING_POLICY:
        print(f"tp_group_size: {config['tp_group_size']}, dp_group_size: {config['dp_group_size']}, pp_group_size: {config['pp_group_size']}, coll_cnt_per_tp_group: {config['coll_cnt_per_tp_group']}, coll_cnt_per_dp_group: {config['coll_cnt_per_dp_group']}", flush=True)


if __name__ == "__main__":
    print("\n\n================\nStart\n================\n\n", flush=True)

    parser = argparse.ArgumentParser(description='GPU deadlock simulator.')
    parser.add_argument('-f', '--file', required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    
    with open(args.file, 'r') as file:
        config = json.load(file)

    sum_deadlock_rounds = Value('i', 0)  # 类型代码（如 'i' 表示整数）
    lock = Lock()

    num_processes = config["num_process"]
    total_rounds = config['total_rounds']

    processes = [Process(target=init_7_main_loop, args=(config, sum_deadlock_rounds, lock)) for _ in range(num_processes)]
    
    for p in processes:
        p.start()
        print(p.pid, flush=True)

    print(f"start {num_processes} processes, each {total_rounds} rounds\n\n", flush=True)

    # 等待所有进程完成
    for p in processes:
        p.join()

    print(f"\n================\nSummary: #Deadlock round: {sum_deadlock_rounds.value} out of {total_rounds * num_processes} rounds, ratio: {sum_deadlock_rounds.value / (total_rounds * num_processes)}", flush=True)
