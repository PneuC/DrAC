import os
import jpype
from math import ceil
from enum import Enum
from root import PRJROOT
from jpype import JString, JInt, JBoolean, JLong
from typing import Union, Dict
from smb.level import MarioLevel, LevelRender
from myutils.filesys import gp

JVMPath = None


class MarioJavaAgents(Enum):
    Runner = 'agents.robinBaumgarten'
    Killer = 'agents.killer'
    Collector = 'agents.collector'

    def __str__(self):
        return self.value + '.Agent'


class MarioProxy:
    def __init__(self):
        if not jpype.isJVMStarted():
            jar_path = gp('smb/Mario-AI-Framework.jar')
            jpype.startJVM(
                jpype.getDefaultJVMPath() if JVMPath is None else JVMPath,
                f"-Djava.class.path={jar_path}", '-Xmx2g'
            )
            """
                -Xmx{size} set the heap size.
            """
        jpype.JClass("java.lang.System").setProperty('user.dir', os.path.join(PRJROOT, 'smb'))
        self.__proxy = jpype.JClass("MarioProxy")()

    @staticmethod
    def __extract_res(jresult):
        return {
            'status': str(jresult.getGameStatus().toString()),
            'completing-ratio': float(jresult.getCompletionPercentage()),
            '#kills': int(jresult.getKillsTotal()),
            '#kills-by-fire': int(jresult.getKillsByFire()),
            '#kills-by-stomp': int(jresult.getKillsByStomp()),
            '#kills-by-shell': int(jresult.getKillsByShell()),
            'trace': [
                [float(item.getMarioX()), float(item.getMarioY())]
                for item in jresult.getAgentEvents()
            ],
            'JAgentEvents': jresult.getAgentEvents()
        }

    def play_game(self, level: Union[str, MarioLevel], lives=0, verbose=False, scale=2):
        if type(level) == str:
            level = MarioLevel.from_file(level)
        jresult = self.__proxy.playGame(JString(str(level)), JInt(lives), JBoolean(verbose), JInt(scale))
        return MarioProxy.__extract_res(jresult)

    def simulate_game(self,
        level: Union[str, MarioLevel],
        agent: MarioJavaAgents=MarioJavaAgents.Runner,
        render: bool=False,
        realTimeLim: int = 0
    ) -> Dict:
        """
        Run simulation with an agent for a given level
        :param level: if type is str, must be path_ of a valid level file.
        :param agent: type of the agent.
        :param render: render or not.
        :param realTimeLim: Real-time limit, in unit of microsecond.
        :return: dictionary of the results.
        """
        # start_time = time.perf_counter()
        jagent = jpype.JClass(str(agent))()
        if type(level) == str:
            level = MarioLevel.from_file(level)
        fps = 24 if render else 0
        jresult = self.__proxy.simulateGame(JString(str(level)), jagent, JBoolean(render), JInt(fps), JLong(realTimeLim * 1000))
        return MarioProxy.__extract_res(jresult)

    def simulate_complete(self,
        level: Union[str, MarioLevel],
        agent: MarioJavaAgents=MarioJavaAgents.Runner,
        segTimeK: int=80
    ) -> Dict:
        ts = LevelRender.tex_size
        jagent = jpype.JClass(str(agent))()
        if type(level) == str:
            level = MarioLevel.from_file(level)
        reached_tile = 0
        res = {'restarts': [], 'trace': []}
        dx = 0
        win = False
        while not win and reached_tile < level.w - 1:
            jresult = self.__proxy.simulateWithSegmentwiseTimeout(
                JString(str(level[:, reached_tile:])), jagent, JInt(segTimeK))
            pyresult = MarioProxy.__extract_res(jresult)
            reached = pyresult['trace'][-1][0]
            reached_tile += ceil(reached / ts)
            if pyresult['status'] != 'WIN':
                res['restarts'].append(reached_tile)
            else:
                win = True
            res['trace'] += [[dx + item[0], item[1]] for item in pyresult['trace']]
            dx = reached_tile * ts
        return res

    @staticmethod
    def get_seg_infos(full_info, check_points=None):
        restarts, trace = full_info['restarts'], full_info['trace']
        W = MarioLevel.seg_width
        ts = LevelRender.tex_size
        if check_points is None:
            end = ceil(trace[-1][0] / ts)
            check_points = [x for x in range(W, end, W)]
            check_points.append(end)
        res = [{'trace': [], 'playable': True} for _ in check_points]
        s, e, i = 0, 0, 0
        restart_pointer = 0
        while True:
            while e < len(trace) and trace[e][0] < ts * check_points[i]:
                if restart_pointer < len(restarts) and restarts[restart_pointer] < check_points[i]:
                    res[i]['playable'] = False
                    restart_pointer += 1
                e += 1
            x0 = trace[s][0]
            res[i]['trace'] = [[item[0] - x0, item[1]] for item in trace[s:e]]
            i += 1
            if i == len(check_points):
                break
            s = e
        return res

if __name__ == '__main__':
    simulator = MarioProxy()
    lvl = MarioLevel.from_file('smb/levels/lvl-1.lvl')
    print(simulator.play_game(lvl))
