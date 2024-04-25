#! /usr/bin/python3

import numpy as np
from typing import overload

SQRT3 = np.sqrt(3)


class RmcSim2d:
    # 格子定数(nm)
    A_MG = 0.321
    # 粒子の半径(nm)
    R_PARTICLE = 0.355
    # 乱数を使った試行回数の上限
    MAX_RAND_TRIAL = 100
    # 粒子の移動方向ごとの変位(格子座標)
    STEP = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))
    # 粒子近傍の禁止域の相対位置
    PROHIBITED = np.array([(
        0, 1, 1,  0,  -1, -1, 0,    # 0,1NN
        2, 1, -1, -2, -1, 1,        # 2NN
        2, 2, 0,  -2, -2, 0,        # 3NN
        3, 3, 2,  1,  -1, -2, -3, -3, -2, -1, 1, 2, # 4NN
        3, 3, 0,  -3, -3, 0         # 5NN
    ), (
        0, 0, 1, 1,  0,  -1, -1,    # 0,1NN
        1, 2, 1, -1, -2, -1,        # 2NN
        0, 2, 2, 0,  -2, -2,        # 3NN
        1, 2, 3, 3,  2,  1,  -1, -2, -3, -3, -2, -1, # 4NN
        0, 3, 3, 0,  -3, -3         # 5NN
)])

    def __init__(self, seed: int = 0):
        # モデル空間のサイズ
        self.La, self.Lb = 0, 0
        # モデル中の粒子数
        self.N = 0
        # フィッティングするqの配列
        self.qx, qy = np.array([]), np.array([])
        # 実験データ
        self.i_exp = np.array([])


        # 粒子の座標(格子座標:a=x, b=-0.5x+rt(3)/2 y)
        self.pos_ab = np.array([[np.nan, np.nan]])
        # 格子ベクトルの定義
        self.set_rot(0)
        # 禁止位置のab座標の配列 : インデックスは[粒子の番号,相対位置]の順
        self.prohibited_a, self.prohibited_b = np.array([]), np.array([])
        # 散乱振幅の実部、虚部
        self.a_re, self.a_im = np.array([]), np.array([])
        # 散乱強度
        self.i_sim = np.array([])
        self.i_sum = np.nan
        self.i_dtype = np.int32

        # 乱数生成器
        self.rs = np.random.RandomState(seed)
        return

    def run(self, n_move: int, max_iter: int,
            sigma2:float=1e-1, thresh: float = 1e-20) -> np.ndarray:
        """フィッティングを実行する

        Parameters
        ----------
        n_move: int
            1ステップで動かす粒子の数
        max_iter : int
            試行回数の上限
        thresh : float, optional
            停止する残差の閾値, by default 1e-20

        Returns
        -------
        np.ndarray
            residualの履歴
        """
        self.compute_i()
        res_hist = np.empty(max_iter + 1)
        res_hist[0] = self.compute_residual()
        sigma2 = sigma2 * res_hist[0]
        ran = self.rs.rand(max_iter)
        for i in range(max_iter):
            before_ab = self.pos_ab.copy()
            before_xy, after_xy = self.__move(n_move)
            self.update_i(before_xy, after_xy)
            new_res = self.compute_residual()
            d_res = new_res - res_hist[i]
            if (d_res<0) or ((np.exp(-d_res / sigma2) > ran[i])):
                pass # accept
            else:
                # rejectとして元に戻す
                self.pos_ab = before_ab
                self.update_i(after_xy, before_xy)
                new_res = res_hist[i]
            res_hist[i + 1] = new_res

            if new_res < thresh:
                res_hist = res_hist[:i + 2]
                break

        self.pos_xy = np.dot(self.pos_ab, self.conv)
        return res_hist

    def set_exp_data(self, data: np.ndarray, qx: np.ndarray, qy: np.ndarray,
                     *, dtype=np.int32, v_min:float=2.0, v_max:float=2**20, q_thresh:float=1.0):
        """散乱強度の実験値をセットする
        散乱強度は全要素の和が1になるように規格化する

        Parameters
        ----------
        data : np.ndarray
            二次元散乱強度プロフファイル
        qx : np.ndarray
            dataの第2軸に対応するqの配列
        qy : np.ndarray
            dataの第1軸に対応するqの配列
        dtype : np.dtype, optional
            散乱強度の型, by default np.int32
        v_min : float, optional
            強度がv_min以下の部分をマスクする, by default 2.0
        v_max : float, optional
            強度がv_max以上の部分をマスクする, by default 2**20
        q_thresh : float, optional
            qがq_thresh以下の部分をマスクする, by default 1.0
        """
        xx, yy = np.meshgrid(qx, qy)
        q = np.sqrt(xx**2 + yy**2)

        # マスクで隠す画素が0
        self.mask = ((data > v_min) * (data < v_max) * (q > q_thresh)).astype(np.int8)
        self.i_exp = (data * self.mask).astype(dtype)
        self.i_sum = np.sum(self.i_exp)
        self.i_dtype = dtype
        self.qx, self.qy = qx, qy

        w = q * self.R_PARTICLE
        self.i_par = (3 * (np.sin(w) - w * np.cos(w)) * w**(-3))**2
        self.i_par[w == 0] = 0
        return

    def set_model(self, La: int, Lb: int, pos_ab: np.ndarray):
        """モデル空間のサイズと粒子数を設定する

        Parameters
        ----------
        La : int
        Lb : int
        pos_ab : np.ndarray
        """
        assert pos_ab.shape[1] == 2
        assert np.all((pos_ab[:0] >= 0) * (pos_ab[:0] < La))
        assert np.all((pos_ab[:1] >= 0) * (pos_ab[:1] < Lb))
        self.La, self.Lb = La, Lb
        self.N = pos_ab.shape[0]
        self.pos_ab = pos_ab.copy()
        self.pos_xy = np.dot(self.pos_ab, self.conv)

        # 禁止位置の生成
        # xx: 各列に相対位置が入った配列
        # yy: 各行に粒子位置が入った配列
        xx, yy = np.meshgrid(self.PROHIBITED[:,0], self.pos_ab[:,0])
        self.prohibited_a = (xx + yy) % self.La
        xx, yy = np.meshgrid(self.PROHIBITED[:,1], self.pos_ab[:,1])
        self.prohibited_b = (xx + yy) % self.Lb
        return

    def anneal(self, n_iter:int=1_000, n_move=1):
        for _ in range(n_iter):
            self.__move(n_move=n_move)
        self.pos_xy = np.dot(self.pos_ab, self.conv)
        return

    def set_rot(self, rot: float):
        """格子ベクトルを開店する

        Parameters
        ----------
        rot : float
            a軸とx軸のなす角[deg]
        """
        ea, eb = np.array([1, 0]), np.array([-0.5, SQRT3/2])
        rot = np.radians(rot)
        rot_mat = np.array([
            [np.cos(rot), -np.sin(rot)],
            [np.sin(rot), np.cos(rot)]
        ])
        self.ea = np.dot(rot_mat, ea) * self.A_MG
        self.eb = np.dot(rot_mat, eb) * self.A_MG
        self.conv = np.vstack((self.ea, self.eb))
        self.pos_xy = np.dot(self.pos_ab, self.conv)
        return

    def compute_i(self):
        """散乱強度を計算する"""
        nx, ny, N = self.qx.size, self.qy.size, self.N
        qx, qy = np.meshgrid(self.qx, self.qy)
        # Kahanの加算アルゴリズム
        a_re, a_im = np.zeros_like(qx), np.zeros_like(qx)
        c_re, c_im = 0.0, 0.0
        for xy in self.pos_xy:
            delta = qx * xy[0] + qy * xy[1] # 位相
            re, im = np.cos(delta), np.sin(delta)
            y_re, y_im = re - c_re, im - c_im
            t_re, t_im = a_re + y_re, a_im + y_im
            c_re, c_im = (t_re - a_re) - y_re, (t_im - a_im) - y_im
            a_re, a_im = t_re, t_im
        self.a_re, self.a_im = a_re, a_im

        i_sim = (self.a_re**2 + self.a_im**2) * self.i_par
        sim_sum = (i_sim * self.mask).sum()
        self.i_sim = (i_sim * self.i_sum / sim_sum).astype(self.i_dtype)
        return

    def update_i(self, before_xy: np.ndarray, after_xy: np.ndarray):
        """直前に動かした粒子の情報を利用して散乱強度を更新する"""
        nx, ny, n_moved = self.qx.size, self.qy.size, before_xy.shape[0]
        qx, qy = np.meshgrid(self.qx, self.qy)
        # Kahanの加算アルゴリズム
        da_re, da_im = np.zeros_like(qx), np.zeros_like(qx) # 前後の振幅変化の総和
        c_re, c_im = 0.0, 0.0
        for before, after in zip(before_xy, after_xy):
            # 前後の位相
            delta_after = qx * after[0] + qy * after[1]
            delta_before = qx * before[0] + qy * before[1]
            # 前後の1粒子毎の振幅変化
            re = np.cos(delta_after) - np.cos(delta_before)
            im = np.sin(delta_after) - np.sin(delta_before)
            y_re, y_im = re - c_re, im - c_im
            t_re, t_im = da_re + y_re, da_im + y_im
            c_re, c_im = (t_re - da_re) - y_re, (t_im - da_im) - y_im
            da_re, da_im = t_re, t_im
        self.a_re += da_re
        self.a_im += da_im

        # 散乱強度(2次元配列)
        i_sim = (self.a_re**2 + self.a_im**2) * self.i_par
        sim_sum = (i_sim * self.mask).sum()
        self.i_sim = (i_sim * self.i_sum / sim_sum).astype(self.i_dtype)
        return

    def compute_residual(self):
        """実験値との残差を計算する"""
        # assert(self.i_sim.sum()/self.i_sum-1 < 1e-100)
        return np.sum((self.i_exp*self.mask - self.i_sim)**2)

    def __move(self, n_move: int, max_iter: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """禁止域に入らないように粒子を動かす
        **pos_xyを更新しない**

        Parameters
        ----------
        n_move : int
            1ステップで動かす粒子の数
        max_iter : int, optional
            禁止域に入らないように粒子を動かす試行回数, by default 100

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (移動前のxy座標の配列), (移動後のxy座標の配列)
        """
        if max_iter == -1:
            max_iter = self.MAX_RAND_TRIAL

        before_ab = self.pos_ab.copy()
        success = False
        idx = self.rs.choice(self.N, n_move, replace=False)
        for _ in range(max_iter):
            _success = False
            for i in idx:
                __success = False
                dirs = np.arange(6)
                self.rs.shuffle(dirs)
                for d in dirs:
                    new_ab = self.pos_ab[i] + self.STEP[d]
                    new_ab[0] = new_ab[0] % self.La
                    new_ab[1] = new_ab[1] % self.Lb
                    # 禁止域に入らないかチェック
                    m = (self.prohibited_a == new_ab[0]) * (self.prohibited_b == new_ab[1])
                    if m.sum() > 1: # 元の場所の近傍以外のprohibitedと一致したらアウト
                        continue
                    self.pos_ab[i] = new_ab
                    __success = True
                    break
                if not __success:
                    break
                _success = True

            # 失敗した場合は元に戻して再試行
            if not _success:
                self.pos_ab[idx] = before_ab[idx]
                idx = self.rs.choice(self.N, n_move, replace=False)
                continue

            success = True
            break

        if not success:
            raise RuntimeError("Failed to move particles")

        # idxには移動した粒子のインデックスが残っている
        before_xy = np.dot(before_ab[idx], self.conv)
        after_xy = np.dot(self.pos_ab[idx], self.conv)
        return before_xy, after_xy

    def move(self, n_move: int, max_iter: int = -1):
        """粒子を動かす

        Parameters
        ----------
        n_move : int
            1ステップで動かす粒子の数
        max_iter : int, optional
            禁止域に入らないように粒子を動かす試行回数, by default 100
        """
        self.__move(n_move, max_iter)
        self.pos_xy = np.dot(self.pos_ab, self.conv)
        return